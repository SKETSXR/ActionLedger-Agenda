# =============================================================================
# Module: topic_generation_agent
# =============================================================================
# Overview
#   Production-ready LangGraph agent that generates a structured set of interview
#   topics from an already-generated summary. It runs a compact ReAct-style loop
#   with Mongo-backed tools and coerces the final assistant content to the
#   typed Pydantic schema: CollectiveInterviewTopicSchema.
#
# Responsibilities
#   - Build a System prompt from prior pipeline output (generated summary +
#     accumulated feedback) and a minimal Human trigger.
#   - Invoke an LLM bound to Mongo tools (via ToolNode) to propose topics.
#   - Convert the final assistant message into the structured output schema.
#   - Validate alignment against the SkillTree (coverage & must-have skills).
#   - Retry the inner generation if the validation conditions are not met.
#
# Data Flow
#   Outer Graph: START ──► topic_generator ──► should_regenerate ──┬─► END (True)
#                                                                 └─► topic_generator (False)
#   Inner Graph (Mongo loop): agent ─► (tools)* ─► respond
#     1) agent: tool-enabled LLM call plans tools, if any
#     2) tools: ToolNode executes planned tool calls
#     3) respond: convert last tool-free assistant message to schema
#
# Reliability & Observability
#   - Timeouts and retries (exponential backoff) for both LLM and tools.
#   - Human-friendly console logs and rotating file logs.
#   - Compact log redaction for large text fields, with safe previews.
#
# Configuration (Environment Variables)
#   TOPIC_AGENT_LOG_DIR, TOPIC_AGENT_LOG_FILE, TOPIC_AGENT_LOG_LEVEL
#   TOPIC_AGENT_LOG_ROTATE_WHEN, TOPIC_AGENT_LOG_ROTATE_INTERVAL, TOPIC_AGENT_LOG_BACKUP_COUNT
#   TOPIC_AGENT_LLM_TIMEOUT_SECONDS, TOPIC_AGENT_LLM_RETRIES, TOPIC_AGENT_LLM_RETRY_BACKOFF_SECONDS
#   TOPIC_AGENT_TOOL_TIMEOUT_SECONDS, TOPIC_AGENT_TOOL_RETRIES, TOPIC_AGENT_TOOL_RETRY_BACKOFF_SECONDS
#   TOPIC_AGENT_TOOL_MAX_WORKERS
# =============================================================================

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from logging.handlers import TimedRotatingFileHandler
from string import Template
from typing import Any, Callable, Coroutine, List, Optional, Sequence

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
# Config (env-overridable)
# ==============================

AGENT_NAME = "topic_generation_agent"

LOG_DIR = os.getenv("TOPIC_AGENT_LOG_DIR", "logs")
LOG_LEVEL = getattr(logging, os.getenv("TOPIC_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FILE = os.getenv("TOPIC_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("TOPIC_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("TOPIC_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("TOPIC_AGENT_LOG_BACKUP_COUNT", "365"))

LLM_TIMEOUT_SECONDS: float = float(os.getenv("TOPIC_AGENT_LLM_TIMEOUT_SECONDS", "90"))
LLM_RETRIES: int = int(os.getenv("TOPIC_AGENT_LLM_RETRIES", "2"))
LLM_BACKOFF_SECONDS: float = float(os.getenv("TOPIC_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

TOOL_TIMEOUT_SECONDS: float = float(os.getenv("TOPIC_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
TOOL_RETRIES: int = int(os.getenv("TOPIC_AGENT_TOOL_RETRIES", "2"))
TOOL_BACKOFF_SECONDS: float = float(os.getenv("TOPIC_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))

# Single shared executor for sync tool calls with timeout
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("TOPIC_AGENT_TOOL_MAX_WORKERS", "8")))

# Global counter for logging retry iterations in the outer graph
topic_retry_counter = 1


# ==============================
# Logging
# ==============================

def build_logger(
    name: str,
    log_dir: str,
    level: int,
    filename: str,
    when: str,
    interval: int,
    backup_count: int,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, filename)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    rotate_file = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8", utc=False
    )
    rotate_file.setLevel(level)
    rotate_file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(console)
    logger.addHandler(rotate_file)
    return logger


logger = build_logger(
    name=AGENT_NAME,
    log_dir=LOG_DIR,
    level=LOG_LEVEL,
    filename=LOG_FILE,
    when=LOG_ROTATE_WHEN,
    interval=LOG_ROTATE_INTERVAL,
    backup_count=LOG_BACKUP_COUNT,
)


def log_info(msg: str) -> None:
    logger.info(msg)


def log_warning(msg: str) -> None:
    logger.warning(msg)


# ---------- config ----------
RAW_TEXT_FIELDS = {
    "question_guidelines", "guidelines", "template", "prompt",
    "policy", "notes", "rubric", "examples", "description_md",
}


# ==============================
# Helpers (redaction + retries)
# ==============================

def _looks_like_json(text: str) -> bool:
    text = text.strip()
    return (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))


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


def _redact(value: Any, *, omit_fields: bool, preview_len: int = 140) -> Any:
    """Redact long raw text fields for compact logging."""
    value = _jsonish(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = k.lower() if isinstance(k, str) else k
            if isinstance(k, str) and key in RAW_TEXT_FIELDS and isinstance(v, str):
                if omit_fields:
                    continue
                head = (v.strip().splitlines() or [""])[0]
                head = head[:preview_len] + ("…" if len(head) > preview_len else "")
                out[k] = f"<{k}: {len(v)} chars — \"{head}\">"
            else:
                out[k] = _redact(v, omit_fields=omit_fields, preview_len=preview_len)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(v, omit_fields=omit_fields, preview_len=preview_len) for v in value]
    return value


def log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    """Log planned tool calls and trailing tool results in a compact, redacted form."""
    if not messages:
        return

    planned = getattr(ai_msg, "tool_calls", None)
    if planned:
        log_info("Tool plan:")
        for tc in planned:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            logger.info(f"  planned -> {name} args={_compact(_redact(_jsonish(args), omit_fields=False))}")

    tool_msgs = []
    i = len(messages) - 1
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        tool_msgs.append(messages[i])
        i -= 1
    if not tool_msgs:
        return

    log_info("Tool results:")
    for tm in tool_msgs:
        content = getattr(tm, "content", None)
        compact = _redact(_jsonish(content), omit_fields=True)
        logger.info(f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_compact(compact)}")


def log_retry(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


async def retry_async_with_backoff(
    op_factory: Callable[[], Coroutine[Any, Any, Any]],
    *,
    retries: int,
    timeout_s: float,
    backoff_base_s: float,
    retry_reason: str,
    iteration_start: int = 1,
) -> Any:
    """Generic async retry helper with timeout + exponential backoff."""
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= retries:
        try:
            return await asyncio.wait_for(op_factory(), timeout=timeout_s)
        except Exception as exc:
            last_exc = exc
            log_retry(retry_reason, iteration_start + attempt, {"error": str(exc)})
            attempt += 1
            if attempt > retries:
                break
            await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


# ======================================================
# Tool wrapper compatible with bind_tools & ToolNode
# ======================================================

class RetryTool(BaseTool):
    """
    Wrap a BaseTool to add timeout + retry for both sync and async execution paths.
    Keeps name/description/args_schema so it remains compatible with bind_tools & ToolNode.
    """

    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff: float = PrivateAttr()

    def __init__(self, inner: BaseTool, *, retries: int, timeout_s: float, backoff_base_s: float) -> None:
        name = getattr(inner, "name", inner.__class__.__name__)
        description = getattr(inner, "description", "") or "Retried tool wrapper"
        args_schema = getattr(inner, "args_schema", None)

        super().__init__(name=name, description=description, args_schema=args_schema)
        self._inner = inner
        self._retries = retries
        self._timeout_s = timeout_s
        self._backoff = backoff_base_s

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Sync path with threadpool timeout + retries."""
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
                log_retry(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                last_exc = exc
                log_retry(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Async path with timeout + retries; falls back to thread executor if tool has only _run."""
        config = kwargs.pop("config", None)

        async def _call_once():
            if hasattr(self._inner, "_arun"):
                return await getattr(self._inner, "_arun")(*args, **{**kwargs, "config": config})
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._inner._run(*args, **{**kwargs, "config": config}))

        return await retry_async_with_backoff(
            _call_once,
            retries=self._retries,
            timeout_s=self._timeout_s,
            backoff_base_s=self._backoff,
            retry_reason=f"tool_async:{self.name}",
        )


# ---------------- Inner ReAct state for Mongo loop ----------------
class _MongoAgentState(MessagesState):
    """State container for the inner Mongo-enabled ReAct loop."""
    final_response: CollectiveInterviewTopicSchema


class TopicGenerationAgent:
    """
    Generate interview topics by running an inner, tool-enabled loop and
    coercing the final assistant message to CollectiveInterviewTopicSchema.
    """

    llm = _llm_client

    # Tools wrapped with retry/timeout
    _RAW_MONGO_TOOLS: List[BaseTool] = get_mongo_tools(llm=llm)
    MONGO_TOOLS: List[BaseTool] = [
        RetryTool(t, retries=TOOL_RETRIES, timeout_s=TOOL_TIMEOUT_SECONDS, backoff_base_s=TOOL_BACKOFF_SECONDS)
        for t in _RAW_MONGO_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(CollectiveInterviewTopicSchema, method="function_calling")

    _compiled_mongo_graph = None  # Compiled inner graph cache

    # ---------- LLM helpers ----------

    @staticmethod
    async def _invoke_agent(messages: Sequence[Any]) -> Any:
        async def _call():
            if hasattr(TopicGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await TopicGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, TopicGenerationAgent._AGENT_MODEL.invoke, messages)

        log_info("Calling LLM (agent)")
        result = await retry_async_with_backoff(
            _call,
            retries=LLM_RETRIES,
            timeout_s=LLM_TIMEOUT_SECONDS,
            backoff_base_s=LLM_BACKOFF_SECONDS,
            retry_reason="llm:agent",
        )
        log_info("LLM (agent) call succeeded")
        return result

    @staticmethod
    async def _invoke_structured(ai_content: str) -> CollectiveInterviewTopicSchema:
        payload = [HumanMessage(content=ai_content)]

        async def _call():
            if hasattr(TopicGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await TopicGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, TopicGenerationAgent._STRUCTURED_MODEL.invoke, payload)

        log_info("Calling LLM (structured)")
        result = await retry_async_with_backoff(
            _call,
            retries=LLM_RETRIES,
            timeout_s=LLM_TIMEOUT_SECONDS,
            backoff_base_s=LLM_BACKOFF_SECONDS,
            retry_reason="llm:structured",
        )
        log_info("LLM (structured) call succeeded")
        return result

    # ---------- Inner graph nodes ----------

    @staticmethod
    async def _agent_node(state: _MongoAgentState):
        """Invoke the tool-enabled model. ToolNode executes tool calls. Returns {"messages": [...]}."""
        log_tool_activity(state["messages"], ai_msg=None)
        ai = await TopicGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _MongoAgentState):
        """Take the last non-tool-call AI message and coerce to schema. Returns {"final_response": obj}."""
        msgs = state["messages"]

        ai_content = None
        for m in reversed(msgs):
            if getattr(m, "type", None) in ("ai", "assistant"):
                if not getattr(m, "tool_calls", None):
                    ai_content = m.content
                    break
        if ai_content is None:
            for m in reversed(msgs):
                if getattr(m, "type", None) in ("ai", "assistant"):
                    ai_content = m.content
                    break
        if ai_content is None:
            ai_content = msgs[-1].content

        final_obj = await TopicGenerationAgent._invoke_structured(ai_content)
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoAgentState):
        """Route to ToolNode if last assistant message has tool_calls; otherwise coerce to schema."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ---------- Compile inner ReAct graph ----------
    @classmethod
    def _get_mongo_graph(cls):
        if cls._compiled_mongo_graph is not None:
            return cls._compiled_mongo_graph

        workflow = StateGraph(_MongoAgentState)
        workflow.add_node("agent", cls._agent_node)
        workflow.add_node("respond", cls._respond_node)
        workflow.add_node("tools", ToolNode(cls.MONGO_TOOLS, tags=["mongo-tools"]))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", cls._should_continue, {"continue": "tools", "respond": "respond"})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)

        cls._compiled_mongo_graph = workflow.compile()
        return cls._compiled_mongo_graph

    # ---------------- Graph node: topic generation ----------------
    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        feedback_text = state.interview_topics_feedback.feedback if state.interview_topics_feedback else ""
        state.interview_topics_feedbacks += f"\n{feedback_text}\n"

        class AtTemplate(Template):
            delimiter = "@"

        tpl = AtTemplate(TOPIC_GENERATION_AGENT_PROMPT)
        sys_content = tpl.substitute(
            generated_summary=state.generated_summary.model_dump_json(),
            interview_topics_feedbacks=state.interview_topics_feedbacks,
            thread_id=state.id,
        )

        messages = [
            SystemMessage(content=sys_content),
            HumanMessage(content="Based on the instructions, please start the process."),
        ]

        graph = TopicGenerationAgent._get_mongo_graph()
        result = await graph.ainvoke({"messages": messages})

        state.interview_topics = result["final_response"]
        log_info("Topic generation completed")
        return state

    # ---------------- Graph router: should regenerate? ----------------
    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Return True when satisfied (END). Return False to retry topic generation.
        Conditions checked:
          - Total questions across topics matches generated_summary.total_questions.
          - All focus skills are valid leaves of the SkillTree.
          - All 'must' priority skill leaves are included among focus areas; if not,
            inject feedback to add them to the last topic and request a retry.
        """
        global topic_retry_counter

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

        focus_area_skills: List[str] = []
        for topic in state.interview_topics.interview_topics:
            for fo in topic.focus_area:
                val = canon(fo.model_dump().get("skill", ""))
                if val and val not in focus_area_skills:
                    focus_area_skills.append(val)

        total_questions = sum(t.total_questions for t in state.interview_topics.interview_topics)
        if total_questions != state.generated_summary.total_questions:
            log_retry(
                "Total questions mismatch",
                topic_retry_counter,
                {"got": total_questions, "target": state.generated_summary.total_questions},
            )
            topic_retry_counter += 1
            return False

        leaf_set = set(all_skill_leaves)
        for s in focus_area_skills:
            if s not in leaf_set:
                log_retry("Invalid focus skill", topic_retry_counter, {"skill": s})
                topic_retry_counter += 1
                return False

        missing_musts = sorted(set(must_skill_leaves) - set(focus_area_skills))
        if missing_musts:
            prev = state.interview_topics_feedback.feedback if state.interview_topics_feedback else ""
            feedback = (
                prev
                + "Please keep the topic set as is irrespective of below instructions:\n"
                f"```\n{state.interview_topics.model_dump()}\n```\n"
                "But add the list of missing `must` priority skills below to the focus areas of the last topic "
                "(General Skill Assessment):\n"
                + ", ".join(missing_musts)
            )
            state.interview_topics_feedback = {"satisfied": False, "feedback": feedback}
            log_retry("Missing MUST skills", topic_retry_counter, {"missing": missing_musts})
            topic_retry_counter += 1
            return False

        return True  # satisfied

    # ---------------- Outer main topic generation graph ----------------
    @staticmethod
    def get_graph(checkpointer=None):
        """
        START -> topic_generator -> (should_regenerate ? END : topic_generator)
        """
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node("topic_generator", TopicGenerationAgent.topic_generator)
        graph_builder.add_edge(START, "topic_generator")
        graph_builder.add_conditional_edges(
            "topic_generator",
            TopicGenerationAgent.should_regenerate,
            {True: END, False: "topic_generator"},
        )
        return graph_builder.compile(checkpointer=checkpointer, name="Topic Generation Agent")


if __name__ == "__main__":
    graph = TopicGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
