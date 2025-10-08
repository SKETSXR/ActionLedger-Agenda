import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from string import Template
from typing import Any, Callable, Coroutine, List, Optional, Sequence

from pydantic import PrivateAttr
from logging.handlers import TimedRotatingFileHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langchain_core.tools import BaseTool

from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import CollectiveInterviewTopicSchema
from ..schema.input_schema import SkillTreeSchema
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from ..model_handling import llm_tg as _llm_client
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

_topic_retry_counter = 1

# ==============================
# Human-style logging
# ==============================

def _build_logger(
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

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Rotating file handler
    fh = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8", utc=False
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


LOGGER = _build_logger(
    name=AGENT_NAME,
    log_dir=LOG_DIR,
    level=LOG_LEVEL,
    filename=LOG_FILE,
    when=LOG_ROTATE_WHEN,
    interval=LOG_ROTATE_INTERVAL,
    backup_count=LOG_BACKUP_COUNT,
)


def log_info(msg: str) -> None:
    LOGGER.info(msg)


def log_warning(msg: str) -> None:
    LOGGER.warning(msg)


def log_error(msg: str) -> None:
    LOGGER.error(msg)


# Helper to pretty-print JSON strings/objects
def _fmt_full(val: Any) -> str:
    """Return full string; pretty-print JSON strings/objects when possible."""
    try:
        # If it's already a dict/list, pretty print
        if isinstance(val, (dict, list)):
            import json as _json
            return _json.dumps(val, ensure_ascii=False, indent=2)
        # If it's a JSON-looking string, parse and pretty print
        if isinstance(val, str):
            s = val.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                import json as _json
                return _json.dumps(_json.loads(s), ensure_ascii=False, indent=2)
        return str(val)
    except Exception:
        return str(val)


# Tool activity, printed in human like style
def log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    if not messages:
        return
    tool_calls = getattr(ai_msg, "tool_calls", None)
    if tool_calls:
        log_info("Tool plan:")
        for tc in tool_calls:
            try:
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args")
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", None)
            except Exception:
                name, args = str(tc), None
            log_info(f"  planned -> {name} args={_fmt_full(args)}")

    i = len(messages) - 1
    any_results = False
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        if not any_results:
            log_info("Tool results:")
            any_results = True
        tm = messages[i]
        log_info(
            f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_fmt_full(getattr(tm, 'content', None))}"
        )
        i -= 1


def log_retry_iteration(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


# ======================================
# Retry / timeout helper for async ops
# ======================================

async def _retry_async(
    op_factory: Callable[[], Coroutine[Any, Any, Any]],
    *,
    retries: int,
    timeout_s: float,
    backoff_base_s: float,
    retry_reason: str,
    iteration_start: int = 1,
) -> Any:
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= retries:
        try:
            return await asyncio.wait_for(op_factory(), timeout=timeout_s)
        except Exception as exc:
            last_exc = exc
            log_retry_iteration(retry_reason, iteration_start + attempt, {"error": str(exc)})
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
    Delegates to an inner BaseTool but adds timeout + retry for both sync and async paths.
    Preserves name/description/args_schema for bind_tools & ToolNode compatibility.
    """

    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff: float = PrivateAttr()

    def __init__(
        self,
        inner: BaseTool,
        *,
        retries: int,
        timeout_s: float,
        backoff_base_s: float,
    ) -> None:
        name = getattr(inner, "name", inner.__class__.__name__)
        description = getattr(inner, "description", "") or "Retried tool wrapper"
        args_schema = getattr(inner, "args_schema", None)

        super().__init__(name=name, description=description, args_schema=args_schema)
        self._inner = inner
        self._retries = retries
        self._timeout_s = timeout_s
        self._backoff = backoff_base_s

    # Forward kw-only config always (StructuredTool expects it)
    def _run(self, *args: Any, **kwargs: Any) -> Any:
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
                log_retry_iteration(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                last_exc = exc
                log_retry_iteration(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)

        async def _call_once():
            if hasattr(self._inner, "_arun"):
                return await getattr(self._inner, "_arun")(*args, **{**kwargs, "config": config})
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._inner._run(*args, **{**kwargs, "config": config}))

        return await _retry_async(
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
    Generates interview topics from a prepared summary by running a small
    ReAct-style tool-using loop (for Mongo tools) and then coercing the final
    assistant content into a typed schema.
    """

    llm_tg = _llm_client

    _RAW_MONGO_TOOLS: List[BaseTool] = get_mongo_tools(llm=llm_tg)  # expected BaseTool instances
    MONGO_TOOLS: List[BaseTool] = [
        RetryTool(t, retries=TOOL_RETRIES, timeout_s=TOOL_TIMEOUT_SECONDS, backoff_base_s=TOOL_BACKOFF_SECONDS)
        for t in _RAW_MONGO_TOOLS
    ]

    _AGENT_MODEL = llm_tg.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_tg.with_structured_output(
        CollectiveInterviewTopicSchema, method="function_calling"
    )

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
        result = await _retry_async(
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
        result = await _retry_async(
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
        """
        Invoke the tool-enabled model. ToolNode executes tool calls.
        Must return {"messages": [...]}
        """
        log_tool_activity(state["messages"], ai_msg=None)
        ai = await TopicGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _MongoAgentState):
        """
        Take the last non-tool-call AI message content and coerce to schema.
        Must return {"final_response": <obj>}
        """
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
        """
        Router: if the last assistant message planned tool calls, continue to tools.
        Otherwise, proceed to respond (coerce to schema).
        """
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
        workflow.add_node("agent", cls._agent_node)     # pass function objects (no calls)
        workflow.add_node("respond", cls._respond_node)
        workflow.add_node("tools", ToolNode(cls.MONGO_TOOLS, tags=["mongo-tools"]))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            cls._should_continue,
            {"continue": "tools", "respond": "respond"},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)

        cls._compiled_mongo_graph = workflow.compile()
        return cls._compiled_mongo_graph

    # ---------------- Graph node: topic generation ----------------
    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        interview_topics_feedback = (
            state.interview_topics_feedback.feedback
            if state.interview_topics_feedback is not None
            else ""
        )
        state.interview_topics_feedbacks += "\n" + interview_topics_feedback + "\n"

        class AtTemplate(Template):
            delimiter = "@"

        tpl = AtTemplate(TOPIC_GENERATION_AGENT_PROMPT)
        content = tpl.substitute(
            generated_summary=state.generated_summary.model_dump_json(),
            interview_topics_feedbacks=state.interview_topics_feedbacks,
            thread_id=state.id,
        )

        messages = [
            SystemMessage(content=content),
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
        global _topic_retry_counter
        def _canon(s: str) -> str:
            return (s or "").strip().lower()

        def _level3_leaves(root: SkillTreeSchema) -> List[SkillTreeSchema]:
            if not getattr(root, "children", None):
                return []
            leaves: List[SkillTreeSchema] = []
            for domain in (root.children or []):
                for leaf in (domain.children or []):
                    if not getattr(leaf, "children", None):
                        leaves.append(leaf)
            return leaves

        def _level3_must_leaves(root: SkillTreeSchema) -> List[SkillTreeSchema]:
            if not getattr(root, "children", None):
                return []
            musts: List[SkillTreeSchema] = []
            for domain in (root.children or []):
                for leaf in (domain.children or []):
                    if not getattr(leaf, "children", None) and getattr(leaf, "priority", None) == "must":
                        musts.append(leaf)
            return musts

        all_skill_leaves = [_canon(leaf.name) for leaf in _level3_leaves(state.skill_tree)]
        skills_priority_must = [_canon(leaf.name) for leaf in _level3_must_leaves(state.skill_tree)]

        focus_area_list: List[str] = []
        for t in state.interview_topics.interview_topics:
            for i in t.focus_area:
                v = _canon(i.model_dump().get("skill", ""))
                if v and v not in focus_area_list:
                    focus_area_list.append(v)

        total_questions_sum = sum(t.total_questions for t in state.interview_topics.interview_topics)
        if total_questions_sum != state.generated_summary.total_questions:
            log_retry_iteration("Total questions mismatch", _topic_retry_counter, {"got": total_questions_sum, "target": state.generated_summary.total_questions})
            _topic_retry_counter += 1
            return False

        leaf_set = set(all_skill_leaves)
        for s in focus_area_list:
            if s not in leaf_set:
                log_retry_iteration("Invalid focus skill", _topic_retry_counter, {"skill": s})
                _topic_retry_counter += 1
                return False

        missing = sorted(set(skills_priority_must) - set(focus_area_list))
        if missing:
            fb_prev = state.interview_topics_feedback.feedback if state.interview_topics_feedback else ""
            fb = (
                fb_prev
                + "Please keep the topic set as is irrespective of below instructions:\n"
                f"```\n{state.interview_topics.model_dump()}\n```\n"
                "But add the list of missing `must` priority skills below to the focus areas of the last topic "
                "(General Skill Assessment):\n"
                + ", ".join(missing)
            )
            state.interview_topics_feedback = {"satisfied": False, "feedback": fb}
            log_retry_iteration("Missing MUST skills", _topic_retry_counter, {"missing": missing})
            _topic_retry_counter += 1
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
