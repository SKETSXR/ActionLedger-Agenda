# =============================================================================
# Module: topic_generation_agent
# =============================================================================
# Purpose
#   Generate a structured set of interview topics from an already-generated
#   summary, via a compact ReAct-style inner loop with Mongo-backed tools.
#   The final assistant content is coerced to CollectiveInterviewTopicSchema.
#
# Responsibilities
#   • Build a System prompt from prior outputs (generated summary + feedback).
#   • Invoke an LLM bound to Mongo tools (ToolNode) to propose topics.
#   • Convert the final assistant message to the typed schema.
#   • Validate alignment against the SkillTree (coverage & must-have skills).
#   • Retry topic generation until validations pass (outer loop).
#
# Data Flow
#   Outer Graph:
#     START ──► topic_generator ──► should_regenerate ──┬─► END (True)
#                                                      └─► topic_generator (False)
#   Inner ReAct Loop:
#     agent (LLM w/ tools) ─► (tools)* ─► respond (coerce to schema)
#
# Reliability & Observability
#   • Timeouts + exponential-backoff retries for LLM and tools.
#   • Rotating file logs + readable console logs.
#   • Optional payload logging: off | summary | full.
#
# Configuration (Environment Variables)
#   TOPIC_AGENT_LOG_DIR, TOPIC_AGENT_LOG_FILE, TOPIC_AGENT_LOG_LEVEL
#   TOPIC_AGENT_LOG_ROTATE_WHEN, TOPIC_AGENT_LOG_ROTATE_INTERVAL, TOPIC_AGENT_LOG_BACKUP_COUNT
#   TOPIC_AGENT_LLM_TIMEOUT_SECONDS, TOPIC_AGENT_LLM_RETRIES, TOPIC_AGENT_LLM_RETRY_BACKOFF_SECONDS
#   TOPIC_AGENT_TOOL_TIMEOUT_SECONDS, TOPIC_AGENT_TOOL_RETRIES, TOPIC_AGENT_TOOL_RETRY_BACKOFF_SECONDS
#   TOPIC_AGENT_TOOL_MAX_WORKERS
#   TOPIC_AGENT_TOOL_LOG_PAYLOAD             off | summary | full
#   TOPIC_AGENT_RESULT_LOG_PAYLOAD           off | summary | full
# =============================================================================

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from string import Template
from typing import Any, Callable, Coroutine, List, Optional, Sequence, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import PrivateAttr

from ..model_handling import llm_tg as _llm_client
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from ..schema.agent_schema import AgentInternalState
from ..schema.input_schema import SkillTreeSchema
from ..schema.output_schema import CollectiveInterviewTopicSchema
from src.mongo_tools import get_mongo_tools


# ==============================
# Configuration
# ==============================

AGENT_NAME = "topic_generation_agent"


@dataclass(frozen=True)
class TopicAgentConfig:
    log_dir: str = os.getenv("TOPIC_AGENT_LOG_DIR", "logs")
    log_file: str = os.getenv("TOPIC_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
    log_level: int = getattr(logging, os.getenv("TOPIC_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_rotate_when: str = os.getenv("TOPIC_AGENT_LOG_ROTATE_WHEN", "midnight")
    log_rotate_interval: int = int(os.getenv("TOPIC_AGENT_LOG_ROTATE_INTERVAL", "1"))
    log_backup_count: int = int(os.getenv("TOPIC_AGENT_LOG_BACKUP_COUNT", "365"))

    llm_timeout_s: float = float(os.getenv("TOPIC_AGENT_LLM_TIMEOUT_SECONDS", "90"))
    llm_retries: int = int(os.getenv("TOPIC_AGENT_LLM_RETRIES", "2"))
    llm_backoff_base_s: float = float(os.getenv("TOPIC_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

    tool_timeout_s: float = float(os.getenv("TOPIC_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
    tool_retries: int = int(os.getenv("TOPIC_AGENT_TOOL_RETRIES", "2"))
    tool_backoff_base_s: float = float(os.getenv("TOPIC_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))
    tool_max_workers: int = int(os.getenv("TOPIC_AGENT_TOOL_MAX_WORKERS", "8"))

    tool_log_payload: str = os.getenv("TOPIC_AGENT_TOOL_LOG_PAYLOAD", "off").strip().lower()
    result_log_payload: str = os.getenv("TOPIC_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()


CFG = TopicAgentConfig()

# Single shared executor for sync tools with timeouts
_EXECUTOR = ThreadPoolExecutor(max_workers=CFG.tool_max_workers)

# Global counter for outer-loop retry logs
_topic_retry_counter = 1


# ==============================
# Logging
# ==============================

def _get_logger() -> logging.Logger:
    logger = logging.getLogger(AGENT_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(CFG.log_level)
    logger.propagate = False

    os.makedirs(CFG.log_dir, exist_ok=True)
    file_path = os.path.join(CFG.log_dir, CFG.log_file)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(CFG.log_level)
    console.setFormatter(fmt)

    rotate_file = TimedRotatingFileHandler(
        file_path,
        when=CFG.log_rotate_when,
        interval=CFG.log_rotate_interval,
        backupCount=CFG.log_backup_count,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    logging.raiseExceptions = False  # never raise on logging I/O in production
    rotate_file.setLevel(CFG.log_level)
    rotate_file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(rotate_file)
    return logger


LOGGER = _get_logger()


def _log_info(msg: str) -> None:
    LOGGER.info(msg)


def _log_warn(msg: str) -> None:
    LOGGER.warning(msg)


# ==============================
# Small JSON / logging helpers
# ==============================

def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))


def _jsonish(value: Any) -> Any:
    if isinstance(value, str) and _looks_like_json(value):
        try:
            return json.loads(value)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    return value


def _compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
    except Exception:
        return str(value)


def _pydantic_to_obj(obj: Any) -> Any:
    # pydantic v2 preferred
    if hasattr(obj, "model_dump_json"):
        try:
            return json.loads(obj.model_dump_json())
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # v1 / plain dict fallback
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


def _summarize_topics(payload: Any) -> str:
    try:
        data = _pydantic_to_obj(payload)
        if isinstance(data, dict):
            topics = None
            if isinstance(data.get("interview_topics"), list):
                topics = data["interview_topics"]
            elif isinstance(data.get("interview_topics"), dict) and isinstance(
                data["interview_topics"].get("interview_topics"), list
            ):
                topics = data["interview_topics"]["interview_topics"]

            if topics is None:
                for k, v in data.items():
                    if isinstance(v, list) and k.lower().startswith("interview"):
                        topics = v
                        break

            names: List[str] = []
            if isinstance(topics, list):
                for t in topics[:8]:
                    nm = None
                    if isinstance(t, dict):
                        nm = t.get("topic") or t.get("name") or t.get("title") or t.get("label")
                    else:
                        try:
                            md = t.model_dump()
                            nm = md.get("topic") or md.get("name") or md.get("title") or md.get("label")
                        except Exception:
                            nm = None
                    if isinstance(nm, str) and nm.strip():
                        names.append(nm.strip())
                suffix = "..." if topics and len(topics) > 8 else ""
                return f"topics.len={len(topics)} names={names}{suffix}"
            return f"keys={list(data.keys())[:8]}"
        if isinstance(data, list):
            return f"list(len={len(data)})"
        return type(data).__name__
    except Exception:
        return "<unavailable>"


def _render_topics_for_log(payload: Any) -> str:
    mode = CFG.result_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        return _summarize_topics(payload)
    return _compact(_pydantic_to_obj(payload))


def _render_tool_payload(payload: Any) -> str:
    mode = CFG.tool_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        try:
            obj = _jsonish(payload)
            if isinstance(obj, dict):
                keys = list(obj.keys())[:6]
                return f"keys={keys} ok={obj.get('ok')} count={obj.get('count')} data={'yes' if 'data' in obj else 'no'}"
            if isinstance(obj, list):
                return f"list(len={len(obj)})"
            return type(obj).__name__
        except Exception:
            return "<unavailable>"
    return _compact(_jsonish(payload))


def _log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    if not messages:
        return

    # Planned tool calls (from the current AI message), if any
    planned = getattr(ai_msg, "tool_calls", None)
    if planned:
        _log_info("Tool plan:")
        for tc in planned:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            LOGGER.info(f"  planned -> {name} args={_render_tool_payload(args)}")

    # Trailing tool results in the message buffer
    tool_msgs: List[Any] = []
    i = len(messages) - 1
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        tool_msgs.append(messages[i])
        i -= 1
    if not tool_msgs:
        return

    _log_info("Tool results:")
    for tm in tool_msgs:
        content = getattr(tm, "content", None)
        tool_id = getattr(tm, "tool_call_id", None)
        LOGGER.info(f"  result -> id={tool_id} data={_render_tool_payload(content)}")


def _log_retry(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    suffix = f" | extra={extra}" if extra else ""
    _log_warn(f"Retry {iteration}: {reason}{suffix}")


# ==============================
# Async retry helper
# ==============================

async def _retry_async_with_backoff(
    op_factory: Callable[[], Coroutine[Any, Any, Any]],
    *,
    retries: int,
    timeout_s: float,
    backoff_base_s: float,
    retry_reason: str,
    iteration_start: int = 1,
) -> Any:
    """Run op_factory with timeout and exponential-backoff retries."""
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= retries:
        try:
            return await asyncio.wait_for(op_factory(), timeout=timeout_s)
        except Exception as exc:
            last_exc = exc
            _log_retry(retry_reason, iteration_start + attempt, {"error": str(exc)})
            attempt += 1
            if attempt > retries:
                break
            await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


# ==============================
# Tool wrapper (timeout + retry)
# ==============================

class RetryTool(BaseTool):
    """
    Wrap a BaseTool with timeout + retries (both sync and async paths).
    Compatible with bind_tools and ToolNode.
    """

    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff_base_s: float = PrivateAttr()

    def __init__(self, inner: BaseTool, *, retries: int, timeout_s: float, backoff_base_s: float) -> None:
        name = getattr(inner, "name", inner.__class__.__name__)
        description = getattr(inner, "description", "") or "Retried tool wrapper"
        args_schema = getattr(inner, "args_schema", None)
        super().__init__(name=name, description=description, args_schema=args_schema)

        self._inner = inner
        self._retries = retries
        self._timeout_s = timeout_s
        self._backoff_base_s = backoff_base_s

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Sync execution via threadpool with timeout + retries."""
        config = kwargs.pop("config", None)

        def _call_once():
            return self._inner._run(*args, **{**kwargs, "config": config})

        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt <= self._retries:
            future = _EXECUTOR.submit(_call_once)
            try:
                return future.result(timeout=self._timeout_s)
            except FuturesTimeout as exc:
                last_exc = exc
                _log_retry(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                last_exc = exc
                _log_retry(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Async execution with timeout + retries. Falls back to sync in executor."""
        config = kwargs.pop("config", None)

        async def _call_once():
            if hasattr(self._inner, "_arun"):
                return await getattr(self._inner, "_arun")(*args, **{**kwargs, "config": config})
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._inner._run(*args, **{**kwargs, "config": config}))

        return await _retry_async_with_backoff(
            _call_once,
            retries=self._retries,
            timeout_s=self._timeout_s,
            backoff_base_s=self._backoff_base_s,
            retry_reason=f"tool_async:{self.name}",
        )


# ==============================
# Inner ReAct loop state
# ==============================

class _MongoAgentState(MessagesState):
    """State container for the inner Mongo-enabled loop."""
    final_response: CollectiveInterviewTopicSchema


# ==============================
# Agent
# ==============================

Msg = Union[SystemMessage, HumanMessage]


class TopicGenerationAgent:
    """
    Generate interview topics using a tool-enabled inner loop, then coerce the
    final assistant message to CollectiveInterviewTopicSchema.
    """

    llm = _llm_client

    # Tools
    _RAW_TOOLS: List[BaseTool] = get_mongo_tools(llm=llm)
    TOOLS: List[BaseTool] = [
        RetryTool(t, retries=CFG.tool_retries, timeout_s=CFG.tool_timeout_s, backoff_base_s=CFG.tool_backoff_base_s)
        for t in _RAW_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(CollectiveInterviewTopicSchema, method="function_calling")

    _compiled_inner_graph = None  # cache

    # ----- LLM invokers -----

    @staticmethod
    async def _invoke_agent(messages: Sequence[Msg]) -> Any:
        async def _call():
            if hasattr(TopicGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await TopicGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, TopicGenerationAgent._AGENT_MODEL.invoke, messages)

        _log_info("Calling LLM (agent)")
        res = await _retry_async_with_backoff(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:agent",
        )
        _log_info("LLM (agent) call succeeded")
        return res

    @staticmethod
    async def _invoke_structured(ai_content: str) -> CollectiveInterviewTopicSchema:
        payload = [HumanMessage(content=ai_content)]

        async def _call():
            if hasattr(TopicGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await TopicGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, TopicGenerationAgent._STRUCTURED_MODEL.invoke, payload)

        _log_info("Calling LLM (structured)")
        res = await _retry_async_with_backoff(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:structured",
        )
        _log_info("LLM (structured) call succeeded")
        return res

    # ----- Inner graph nodes -----

    @staticmethod
    async def _agent_node(state: _MongoAgentState):
        """Invoke the tool-enabled model. Returns {'messages': [ai_message]}."""
        _log_tool_activity(state["messages"], ai_msg=None)
        ai = await TopicGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _MongoAgentState):
        """Coerce the last tool-free assistant message to the schema."""
        msgs = state["messages"]

        ai_content: Optional[str] = None
        # Prefer last assistant message without tool calls
        for m in reversed(msgs):
            if getattr(m, "type", None) in ("ai", "assistant") and not getattr(m, "tool_calls", None):
                ai_content = m.content
                break
        # Fallback: any assistant message
        if ai_content is None:
            for m in reversed(msgs):
                if getattr(m, "type", None) in ("ai", "assistant"):
                    ai_content = m.content
                    break
        # Absolute fallback: last message content
        if ai_content is None:
            ai_content = msgs[-1].content

        final_obj = await TopicGenerationAgent._invoke_structured(ai_content)
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoAgentState):
        """Route: if last assistant asked for tools, go to ToolNode; else respond."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            _log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ----- Compile inner ReAct graph -----

    @classmethod
    def _get_inner_graph(cls):
        if cls._compiled_inner_graph is not None:
            return cls._compiled_inner_graph

        g = StateGraph(_MongoAgentState)
        g.add_node("agent", cls._agent_node)
        g.add_node("respond", cls._respond_node)
        g.add_node("tools", ToolNode(cls.TOOLS, tags=["mongo-tools"]))
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", cls._should_continue, {"continue": "tools", "respond": "respond"})
        g.add_edge("tools", "agent")
        g.add_edge("respond", END)

        cls._compiled_inner_graph = g.compile()
        return cls._compiled_inner_graph

    # ==========================
    # Outer graph: main node
    # ==========================

    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        # Append latest feedback (if any) into the cumulative feedback string
        feedback_text = state.interview_topics_feedback.feedback if state.interview_topics_feedback else ""
        if feedback_text:
            state.interview_topics_feedbacks += f"\n{feedback_text}\n"

        class AtTemplate(Template):
            delimiter = "@"

        # Render System prompt with @placeholders
        sys_content = AtTemplate(TOPIC_GENERATION_AGENT_PROMPT).substitute(
            generated_summary=state.generated_summary.model_dump_json(),
            interview_topics_feedbacks=state.interview_topics_feedbacks,
            thread_id=state.id,
        )

        messages: List[Msg] = [
            SystemMessage(content=sys_content),
            HumanMessage(content="Based on the instructions, please start the process."),
        ]

        inner_graph = TopicGenerationAgent._get_inner_graph()
        result = await inner_graph.ainvoke({"messages": messages})

        state.interview_topics = result["final_response"]

        # Log generated topics as per toggle
        rendered = _render_topics_for_log(state.interview_topics.model_dump_json(indent=2))
        _log_info(f"Topic generation completed | output={rendered}")

        return state

    # ==========================
    # Outer graph: router
    # ==========================

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Return True when satisfied (END). Return False to retry topic generation.

        Checks:
          • Sum of topic.total_questions equals generated_summary.total_questions.
          • Every focus_area.skill is a valid leaf in the SkillTree (level-3).
          • All 'must' priority leaf skills appear in the focus areas; otherwise
            inject feedback onto the state and request a retry.
        """
        global _topic_retry_counter

        def canon(text: str) -> str:
            return (text or "").strip().lower()

        def level3_leaves(root: SkillTreeSchema) -> List[SkillTreeSchema]:
            if not getattr(root, "children", None):
                return []
            leaves: List[SkillTreeSchema] = []
            for domain in (root.children or []):
                for leaf in (domain.children or []):
                    if not getattr(leaf, "children", None):
                        leaves.append(leaf)
            return leaves

        def level3_must_leaves(root: SkillTreeSchema) -> List[SkillTreeSchema]:
            if not getattr(root, "children", None):
                return []
            musts: List[SkillTreeSchema] = []
            for domain in (root.children or []):
                for leaf in (domain.children or []):
                    if not getattr(leaf, "children", None) and getattr(leaf, "priority", None) == "must":
                        musts.append(leaf)
            return musts

        all_skill_leaves = [canon(leaf.name) for leaf in level3_leaves(state.skill_tree)]
        must_skill_leaves = [canon(leaf.name) for leaf in level3_must_leaves(state.skill_tree)]

        # Collect unique focus-area skills across topics
        focus_area_skills: List[str] = []
        for topic in state.interview_topics.interview_topics:
            for fa in topic.focus_area:
                val = canon(fa.model_dump().get("skill", ""))
                if val and val not in focus_area_skills:
                    focus_area_skills.append(val)

        # 1) Question count must match
        total_questions = sum(t.total_questions for t in state.interview_topics.interview_topics)
        if total_questions != state.generated_summary.total_questions:
            _log_retry(
                "Total questions mismatch",
                _topic_retry_counter,
                {"got": total_questions, "target": state.generated_summary.total_questions},
            )
            _topic_retry_counter += 1
            return False

        # 2) Every focus skill must be a valid leaf
        leaf_set = set(all_skill_leaves)
        for s in focus_area_skills:
            if s not in leaf_set:
                _log_retry("Invalid focus skill", _topic_retry_counter, {"skill": s})
                _topic_retry_counter += 1
                return False

        # 3) All MUST leaves must be present
        missing_musts = sorted(set(must_skill_leaves) - set(focus_area_skills))
        if missing_musts:
            feedback = (
                "<Please keep this topic set as is irrespective of other instructions apart from this feedback ones:\n"
                f"```\n{state.interview_topics.model_dump()}\n```\n"
                "But add this list of missing `must` priority skills as given below to the focus areas of the last topic being (General Skill Assessment):\n"
                + ", ".join(missing_musts) + ">"          
            )
            state.interview_topics_feedback = {"satisfied": False, "feedback": feedback}
            _log_retry("Missing MUST skills", _topic_retry_counter, {"missing": missing_musts})
            _topic_retry_counter += 1
            return False

        return True  # satisfied

    # ==========================
    # Outer graph: builder
    # ==========================

    @staticmethod
    def get_graph(checkpointer=None):
        """
        START -> topic_generator -> (should_regenerate ? END : topic_generator)
        """
        g = StateGraph(state_schema=AgentInternalState)
        g.add_node("topic_generator", TopicGenerationAgent.topic_generator)
        g.add_edge(START, "topic_generator")
        g.add_conditional_edges(
            "topic_generator",
            TopicGenerationAgent.should_regenerate,
            {True: END, False: "topic_generator"},
        )
        return g.compile(checkpointer=checkpointer, name="Topic Generation Agent")


if __name__ == "__main__":
    graph = TopicGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
