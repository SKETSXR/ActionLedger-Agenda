# =============================================================================
# Module: nodes_generation_agent
# =============================================================================
# Overview
#   Production-ready LangGraph agent that builds per-topic “nodes” (question
#   nodes, deep-dive nodes, etc.) from previously generated per-topic discussion
#   summaries. It runs a compact ReAct-style inner loop with Mongo-backed tools
#   and coerces the final assistant content into typed schemas:
#     - TopicWithNodesSchema (per topic)
#     - NodesSchema (container across topics)
#
# Responsibilities
#   - For each (topic, per-topic summary) pair, construct a System prompt.
#   - Invoke an LLM bound to Mongo tools (via ToolNode) in an inner ReAct loop.
#   - Convert the final tool-free assistant message to TopicWithNodesSchema.
#   - Aggregate all topic results into a NodesSchema on the shared state.
#   - Validate container + per-topic schemas; retry if invalid.
#
# Data Flow
#   Outer Graph:
#     START ──► nodes_generator
#                ├─► nodes_generator (should_regenerate=True)
#                └─► END              (should_regenerate=False)
#
#   Inner Graph (per topic):
#     agent ─► (tools)* ─► respond
#       1) agent: tool-enabled LLM plans tools if needed
#       2) tools: ToolNode executes planned tool calls
#       3) respond: coerce last tool-free assistant message to schema
#
# Reliability & Observability
#   - Timeouts + retries (exponential backoff) for both LLM and tools.
#   - Console + rotating file logs with compact redaction for large fields.
#
# Configuration (Environment Variables)
#   NODES_AGENT_LOG_DIR, NODES_AGENT_LOG_FILE, NODES_AGENT_LOG_LEVEL,
#   NODES_AGENT_LOG_ROTATE_WHEN, NODES_AGENT_LOG_ROTATE_INTERVAL,
#   NODES_AGENT_LOG_BACKUP_COUNT,
#   NODES_AGENT_LLM_TIMEOUT_SECONDS, NODES_AGENT_LLM_RETRIES,
#   NODES_AGENT_LLM_RETRY_BACKOFF_SECONDS,
#   NODES_AGENT_TOOL_TIMEOUT_SECONDS, NODES_AGENT_TOOL_RETRIES,
#   NODES_AGENT_TOOL_RETRY_BACKOFF_SECONDS, NODES_AGENT_TOOL_MAX_WORKERS
# =============================================================================

import json
import copy
import logging
import os
import sys
import time
import asyncio
from typing import List, Any, Dict, Optional, Sequence, Callable, Coroutine
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from logging.handlers import TimedRotatingFileHandler
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError, PrivateAttr
from string import Template

from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import NodesSchema, TopicWithNodesSchema
from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
from ..model_handling import llm_n as _llm_client


# ==============================
# Config (env-overridable)
# ==============================

AGENT_NAME = "nodes_agent"

LOG_DIR = os.getenv("NODES_AGENT_LOG_DIR", "logs")
LOG_LEVEL = getattr(logging, os.getenv("NODES_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FILE = os.getenv("NODES_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("NODES_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("NODES_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("NODES_AGENT_LOG_BACKUP_COUNT", "365"))

# --- richer preview knobs for 'question_guidelines' ---
SHOW_FULL_TEXT = os.getenv("QA_LOG_SHOW_FULL_TEXT", "0") == "1"
SHOW_FULL_FIELDS = {
    k.strip().lower()
    for k in os.getenv("QA_LOG_SHOW_FULL_FIELDS", "").split(",")
    if k.strip()
}
QGUIDE_PREVIEW_LEN = int(os.getenv("QA_LOG_QGUIDE_PREVIEW_LEN", "280"))      # chars
QGUIDE_PREVIEW_LINES = int(os.getenv("QA_LOG_QGUIDE_PREVIEW_LINES", "2"))    # lines

# Retry/timeout knobs (namespaced for this agent)
LLM_TIMEOUT_SECONDS: float = float(os.getenv("NODES_AGENT_LLM_TIMEOUT_SECONDS", "90"))
LLM_RETRIES: int = int(os.getenv("NODES_AGENT_LLM_RETRIES", "2"))
LLM_BACKOFF_SECONDS: float = float(os.getenv("NODES_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

TOOL_TIMEOUT_SECONDS: float = float(os.getenv("NODES_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
TOOL_RETRIES: int = int(os.getenv("NODES_AGENT_TOOL_RETRIES", "2"))
TOOL_BACKOFF_SECONDS: float = float(os.getenv("NODES_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))

_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("NODES_AGENT_TOOL_MAX_WORKERS", "8")))

RAW_TEXT_FIELDS = {
    "question_guidelines", "guidelines", "template", "prompt",
    "policy", "notes", "rubric", "examples", "description_md",
}

# Global retry counter used only for logging iteration counts.
nodes_retry_counter = 1


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

    rotating_file = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8", utc=False
    )
    rotating_file.setLevel(level)
    rotating_file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(console)
    logger.addHandler(rotating_file)
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


def log_info(message: str) -> None:
    logger.info(message)


def log_warning(message: str) -> None:
    logger.warning(message)


# ==============================
# Helpers (redaction + compact)
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
    """Redact long raw text fields for compact logging; always show a richer preview for 'question_guidelines'."""
    value = _jsonish(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = k.lower() if isinstance(k, str) else k

            if isinstance(k, str) and key in RAW_TEXT_FIELDS and isinstance(v, str):
                # Show full only if explicitly allowed
                if SHOW_FULL_TEXT or (SHOW_FULL_FIELDS and key in SHOW_FULL_FIELDS):
                    out[k] = v
                    continue

                # Special-case: ALWAYS include a compact preview for question_guidelines,
                # even if omit_fields=True (other raw fields will be dropped).
                if key == "question_guidelines":
                    lines = [ln.strip() for ln in v.strip().splitlines() if ln.strip()]
                    head = " ".join(lines[:max(1, QGUIDE_PREVIEW_LINES)]) or ""
                    if len(head) > QGUIDE_PREVIEW_LEN:
                        head = head[:QGUIDE_PREVIEW_LEN].rstrip() + "…"
                    out[k + "_preview"] = head
                    out[k + "_len"] = len(v)
                    continue

                # All other long text fields:
                if omit_fields:
                    continue
                head = (v.strip().splitlines() or [""])[0]
                if len(head) > preview_len:
                    head = head[:preview_len].rstrip() + "…"
                out[k + "_preview"] = head
                out[k + "_len"] = len(v)
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

    # Gather trailing tool results
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
        logger.info(f"  result <- id={getattr(tm, 'tool_call_id', None)} data={_compact(compact)}")


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


# ---------- Inner ReAct state for Mongo loop (per-topic) ----------
class _MongoNodesState(MessagesState):
    """State container for the inner ReAct loop that generates nodes per topic."""
    final_response: TopicWithNodesSchema


class NodesGenerationAgent:
    """
    Generates per-topic node structures by running a tool-using inner ReAct loop
    (with Mongo tools), coercing the final assistant content into
    TopicWithNodesSchema, and validating the overall NodesSchema container.
    """

    # Bind the imported client under a clearer name for this class.
    llm = _llm_client

    # Wrap Mongo tools with retry/timeout
    _RAW_MONGO_TOOLS: List[BaseTool] = get_mongo_tools(llm=llm)
    MONGO_TOOLS: List[BaseTool] = [
        RetryTool(
            t,
            retries=TOOL_RETRIES,
            timeout_s=TOOL_TIMEOUT_SECONDS,
            backoff_base_s=TOOL_BACKOFF_SECONDS,
        )
        for t in _RAW_MONGO_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(TopicWithNodesSchema, method="function_calling")

    _compiled_graph = None  # cache inner graph

    # ---------- LLM helpers with retries ----------

    @staticmethod
    async def _invoke_agent(messages: Sequence[Any]) -> Any:
        async def _call():
            if hasattr(NodesGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await NodesGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, NodesGenerationAgent._AGENT_MODEL.invoke, messages)

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
    async def _invoke_structured(ai_content: str) -> TopicWithNodesSchema:
        payload = [HumanMessage(content=ai_content)]

        async def _call():
            if hasattr(NodesGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await NodesGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, NodesGenerationAgent._STRUCTURED_MODEL.invoke, payload)

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

    # ---------- Async node impls ----------
    @staticmethod
    async def _agent_node(state: _MongoNodesState):
        log_tool_activity(state["messages"], ai_msg=None)
        ai = await NodesGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _MongoNodesState):
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

        final_obj = await NodesGenerationAgent._invoke_structured(ai_content)
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoNodesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ---------- RunnableLambda wrappers (avoid coroutine return) ----------
    @staticmethod
    async def _agent_node_async(state: _MongoNodesState):
        return await NodesGenerationAgent._agent_node(state)

    @staticmethod
    async def _respond_node_async(state: _MongoNodesState):
        return await NodesGenerationAgent._respond_node(state)

    # ---------- Compile inner graph ----------
    @classmethod
    def _get_inner_graph(cls):
        if cls._compiled_graph is not None:
            return cls._compiled_graph

        workflow = StateGraph(_MongoNodesState)
        workflow.add_node("agent", RunnableLambda(cls._agent_node_async))
        workflow.add_node("respond", RunnableLambda(cls._respond_node_async))
        workflow.add_node("tools", ToolNode(cls.MONGO_TOOLS, tags=["mongo-tools+arith-tools"]))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            cls._should_continue,
            {"continue": "tools", "respond": "respond"},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)
        cls._compiled_graph = workflow.compile()
        return cls._compiled_graph

    # ----------------- utilities -----------------
    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "dict"):
            return x.dict()
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _get_topic_name(obj: Any) -> str:
        d = NodesGenerationAgent._as_dict(obj)
        for k in ("topic", "name", "title", "label"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    @staticmethod
    def _to_primitive(x: Any) -> Any:
        if hasattr(x, "model_dump"):
            try:
                return NodesGenerationAgent._to_primitive(x.model_dump())
            except Exception:
                pass
        if hasattr(x, "dict"):
            try:
                return NodesGenerationAgent._to_primitive(x.dict())
            except Exception:
                pass
        if isinstance(x, dict):
            return {k: NodesGenerationAgent._to_primitive(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [NodesGenerationAgent._to_primitive(v) for v in x]
        if isinstance(x, (datetime, date)):
            return x.isoformat()
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if hasattr(x, "__dict__"):
            return NodesGenerationAgent._to_primitive(vars(x))
        return str(x)

    @staticmethod
    def _to_json_one(x: Any) -> str:
        return json.dumps(NodesGenerationAgent._to_primitive(copy.deepcopy(x)), ensure_ascii=False)

    @staticmethod
    async def _gen_once(
        per_topic_summary_json: str,
        thread_id: str,
        nodes_error: str,
    ) -> TopicWithNodesSchema:
        class AtTemplate(Template):
            delimiter = "@"

        tpl = AtTemplate(NODES_AGENT_PROMPT)
        content = tpl.substitute(
            per_topic_summary_json=per_topic_summary_json,
            thread_id=thread_id,
            nodes_error=nodes_error,
        )

        sys_message = SystemMessage(content=content)
        trigger_message = HumanMessage(content="Based on the provided instructions please start the process")

        graph = NodesGenerationAgent._get_inner_graph()
        result = await graph.ainvoke({"messages": [sys_message, trigger_message]})
        return result["final_response"]

    # ----------------------------- Main node generation -----------------------------
    @staticmethod
    async def nodes_generator(state: AgentInternalState) -> AgentInternalState:
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")

        topics_list = list(state.interview_topics.interview_topics)
        try:
            summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
        except Exception:
            summaries_list = list(state.discussion_summary_per_topic)

        pair_count = min(len(topics_list), len(summaries_list))
        if pair_count == 0:
            raise ValueError("No topics/summaries to process.")

        snapshot = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True,
        )

        topic_with_nodes_list: List[TopicWithNodesSchema] = []

        for dspt_obj in zip(topics_list, summaries_list):
            per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)
            resp = await NodesGenerationAgent._gen_once(
                per_topic_summary_json, state.id, state.nodes_error
            )
            topic_with_nodes_list.append(resp)

        after = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True,
        )
        if after != snapshot:
            raise RuntimeError("discussion_summary_per_topic mutated during node generation")

        state.nodes = NodesSchema(
            topics_with_nodes=[
                t.model_dump() if hasattr(t, "model_dump") else t for t in topic_with_nodes_list
            ]
        )
        log_info("Nodes generation completed")
        return state

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        global nodes_retry_counter

        if getattr(state, "nodes", None) is None:
            return True

        try:
            NodesSchema.model_validate(
                state.nodes.model_dump() if hasattr(state.nodes, "model_dump") else state.nodes
            )
        except ValidationError as ve:
            state.nodes_error += (
                "The previous generated o/p did not follow the given schema as it got following errors:\n"
                + (getattr(state, "nodes_error", "") or "")
                + "\n[NodesSchema ValidationError]\n"
                + str(ve)
                + "\n"
            )
            log_retry_iteration(
                reason=f"[NodesGen][ValidationError] Container NodesSchema invalid\n {ve}",
                iteration=nodes_retry_counter,
            )
            nodes_retry_counter += 1
            return True

        try:
            topics_payload = (
                state.nodes.topics_with_nodes
                if hasattr(state.nodes, "topics_with_nodes")
                else state.nodes.get("topics_with_nodes", [])
            )
        except Exception as e:
            state.nodes_error += "\n[NodesSchema Payload Error]\n" + str(e) + "\n"
            log_retry_iteration(
                reason=f"[NodesGen][ValidationError] Could not read topics_with_nodes: {e}",
                iteration=nodes_retry_counter,
            )
            nodes_retry_counter += 1
            return True

        any_invalid = False
        for idx, item in enumerate(topics_payload):
            try:
                TopicWithNodesSchema.model_validate(
                    item.model_dump() if hasattr(item, "model_dump") else item
                )
            except ValidationError as ve:
                any_invalid = True
                state.nodes_error += f"\n[TopicWithNodesSchema ValidationError idx={idx}]\n{ve}\n"

        if any_invalid:
            log_retry_iteration("node schema error", iteration=nodes_retry_counter)
            nodes_retry_counter += 1
        return any_invalid

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Graph for node generation:
        START -> nodes_generator -> (should_regenerate ? nodes_generator : END)
        """
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
        gb.add_edge(START, "nodes_generator")
        gb.add_conditional_edges(
            "nodes_generator",
            NodesGenerationAgent.should_regenerate,
            {True: "nodes_generator", False: END},
        )
        return gb.compile(checkpointer=checkpointer, name="Nodes Generation Agent")


if __name__ == "__main__":
    graph = NodesGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
