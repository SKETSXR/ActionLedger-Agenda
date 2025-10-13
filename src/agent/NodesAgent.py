# =============================================================================
# Module: nodes_generation_agent
# =============================================================================
# Purpose
#   Build per-topic “nodes” (question/deep-dive/etc.) from previously generated
#   per-topic discussion summaries via a compact ReAct-style inner loop that
#   uses Mongo-backed tools. The final assistant message is coerced to:
#     • TopicWithNodesSchema (per topic)
#     • NodesSchema (container across topics)
#
# Responsibilities
#   • For each (topic, per-topic summary) pair, render a System prompt.
#   • Invoke an LLM bound to Mongo tools (ToolNode) inside an inner ReAct loop.
#   • Convert the last tool-free assistant message to TopicWithNodesSchema.
#   • Aggregate all topic results into a NodesSchema on shared state.
#   • Validate container and per-topic schemas; retry if invalid.
#
# Data Flow
#   Outer Graph:
#     START ──► nodes_generator
#                ├─► nodes_generator (should_regenerate=True)
#                └─► END              (should_regenerate=False)
#
#   Inner Graph (per topic):
#     agent (LLM w/ tools) ─► (tools)* ─► respond (coerce to schema)
#
# Reliability & Observability
#   • Timeouts + exponential-backoff retries for LLM and tools.
#   • Console + rotating file logs.
#   • Optional payload logging: tools and final result (off | summary | full).
#
# Configuration (Environment Variables)
#   NODES_AGENT_LOG_DIR, NODES_AGENT_LOG_FILE, NODES_AGENT_LOG_LEVEL
#   NODES_AGENT_LOG_ROTATE_WHEN, NODES_AGENT_LOG_ROTATE_INTERVAL, NODES_AGENT_LOG_BACKUP_COUNT
#   NODES_AGENT_LLM_TIMEOUT_SECONDS, NODES_AGENT_LLM_RETRIES, NODES_AGENT_LLM_RETRY_BACKOFF_SECONDS
#   NODES_AGENT_TOOL_TIMEOUT_SECONDS, NODES_AGENT_TOOL_RETRIES, NODES_AGENT_TOOL_RETRY_BACKOFF_SECONDS
#   NODES_AGENT_TOOL_MAX_WORKERS
#   NODES_AGENT_TOOL_LOG_PAYLOAD            off | summary | full
#   NODES_AGENT_RESULT_LOG_PAYLOAD          off | summary | full
# =============================================================================

import asyncio
import copy
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import date, datetime
from logging.handlers import TimedRotatingFileHandler
from string import Template
from typing import Any, Dict, List, Optional, Sequence, Callable, Coroutine

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import PrivateAttr, ValidationError

from src.mongo_tools import get_mongo_tools
from ..model_handling import llm_n as _llm_client
from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import NodesSchema, TopicWithNodesSchema


# ==============================
# Configuration
# ==============================

AGENT_NAME = "nodes_agent"


@dataclass(frozen=True)
class NodesAgentConfig:
    log_dir: str = os.getenv("NODES_AGENT_LOG_DIR", "logs")
    log_file: str = os.getenv("NODES_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
    log_level: int = getattr(logging, os.getenv("NODES_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_rotate_when: str = os.getenv("NODES_AGENT_LOG_ROTATE_WHEN", "midnight")
    log_rotate_interval: int = int(os.getenv("NODES_AGENT_LOG_ROTATE_INTERVAL", "1"))
    log_backup_count: int = int(os.getenv("NODES_AGENT_LOG_BACKUP_COUNT", "365"))

    llm_timeout_s: float = float(os.getenv("NODES_AGENT_LLM_TIMEOUT_SECONDS", "90"))
    llm_retries: int = int(os.getenv("NODES_AGENT_LLM_RETRIES", "2"))
    llm_backoff_base_s: float = float(os.getenv("NODES_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

    tool_timeout_s: float = float(os.getenv("NODES_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
    tool_retries: int = int(os.getenv("NODES_AGENT_TOOL_RETRIES", "2"))
    tool_backoff_base_s: float = float(os.getenv("NODES_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))
    tool_max_workers: int = int(os.getenv("NODES_AGENT_TOOL_MAX_WORKERS", "8"))

    tool_log_payload: str = os.getenv("NODES_AGENT_TOOL_LOG_PAYLOAD", "off").strip().lower()
    result_log_payload: str = os.getenv("NODES_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()


CFG = NodesAgentConfig()

# Shared executor for sync tool calls with timeouts
_EXECUTOR = ThreadPoolExecutor(max_workers=CFG.tool_max_workers)

# Global retry counter (logging only)
_nodes_retry_counter = 1


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

    rotating_file = TimedRotatingFileHandler(
        file_path,
        when=CFG.log_rotate_when,
        interval=CFG.log_rotate_interval,
        backupCount=CFG.log_backup_count,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    logging.raiseExceptions = False  # never raise on logging I/O in production
    rotating_file.setLevel(CFG.log_level)
    rotating_file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(rotating_file)
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
    t = (text or "").strip()
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
    # pydantic v1 / dict-like
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


def _summarize_tool_payload(payload: Any) -> str:
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


def _render_tool_payload(payload: Any) -> str:
    mode = CFG.tool_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        return _summarize_tool_payload(payload)
    return _compact(_jsonish(payload))


def _summarize_nodes_result(payload: Any) -> str:
    """
    For NodesSchema:
      • total topics
      • first few topic names
      • node-counts for the first few topics (len only)
    """
    try:
        data = _pydantic_to_obj(payload)
        topics = data.get("topics_with_nodes") if isinstance(data, dict) else getattr(payload, "topics_with_nodes", None)

        if isinstance(topics, list):
            n = len(topics)
            names: List[str] = []
            counts: List[int] = []
            for t in topics[:6]:
                td = _pydantic_to_obj(t)
                nm = None
                if isinstance(td, dict):
                    nm = td.get("topic") or td.get("name") or td.get("title") or td.get("label")
                    nodes = td.get("nodes")
                else:
                    nm = getattr(td, "topic", None) or getattr(td, "name", None) or getattr(td, "title", None) or getattr(td, "label", None)
                    nodes = getattr(td, "nodes", None)
                names.append((nm or "Unknown").strip() if isinstance(nm, str) else "Unknown")
                counts.append(len(nodes) if isinstance(nodes, list) else 0)
            more = "..." if n > 6 else ""
            return f"topics={n} names={names}{more} node_counts_first={counts}"

        if isinstance(data, dict):
            return f"keys={list(data.keys())[:8]}"
        if isinstance(data, list):
            return f"list(len={len(data)})"
        return type(data).__name__
    except Exception:
        return "<unavailable>"


def _render_nodes_result(payload: Any) -> str:
    mode = CFG.result_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        return _summarize_nodes_result(payload)
    return _compact(_pydantic_to_obj(payload))


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
        LOGGER.info(f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_render_tool_payload(content)}")


def _log_retry(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    suffix = f" | extra={extra}" if extra else ""
    _log_warn(f"Retry {iteration}: {reason}{suffix}")


# ==============================
# Async retry helper
# ==============================

async def _retry_async(
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
    while attempt <= CFG.llm_retries if "llm" in retry_reason else attempt <= CFG.tool_retries:
        try:
            return await asyncio.wait_for(op_factory(), timeout=timeout_s)
        except Exception as exc:
            last_exc = exc
            _log_retry(retry_reason, iteration_start + attempt, {"error": str(exc)})
            attempt += 1
            if ("llm" in retry_reason and attempt > CFG.llm_retries) or ("tool" in retry_reason and attempt > CFG.tool_retries):
                break
            backoff = (CFG.llm_backoff_base_s if "llm" in retry_reason else CFG.tool_backoff_base_s) * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)
    assert last_exc is not None
    raise last_exc


# ==============================
# Tool wrapper (timeout + retry)
# ==============================

class RetryTool(BaseTool):
    """
    Wrap a BaseTool with timeout + retries for both sync and async paths.
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
        """Sync path via threadpool with timeout + retries."""
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
        """Async path with timeout + retries; falls back to sync executor."""
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
            backoff_base_s=self._backoff_base_s,
            retry_reason=f"tool_async:{self.name}",
        )


# ==============================
# Inner ReAct loop (per topic)
# ==============================

class _MongoNodesState(MessagesState):
    """State container for the inner ReAct loop that generates nodes per topic."""
    final_response: TopicWithNodesSchema


class NodesGenerationAgent:
    """
    Generate per-topic node structures via a tool-enabled inner loop, coerce to
    TopicWithNodesSchema, and validate the final NodesSchema container.
    """

    llm = _llm_client

    # Tools (wrapped with retry/timeout)
    _RAW_TOOLS: List[BaseTool] = get_mongo_tools(llm=llm)
    TOOLS: List[BaseTool] = [
        RetryTool(t, retries=CFG.tool_retries, timeout_s=CFG.tool_timeout_s, backoff_base_s=CFG.tool_backoff_base_s)
        for t in _RAW_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(TopicWithNodesSchema, method="function_calling")

    _compiled_graph = None  # cache

    # ----- LLM invokers -----

    @staticmethod
    async def _invoke_agent(messages: Sequence[Any]) -> Any:
        async def _call():
            if hasattr(NodesGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await NodesGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, NodesGenerationAgent._AGENT_MODEL.invoke, messages)

        _log_info("Calling LLM (agent)")
        res = await _retry_async(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:agent",
        )
        _log_info("LLM (agent) call succeeded")
        return res

    @staticmethod
    async def _invoke_structured(ai_content: str) -> TopicWithNodesSchema:
        payload = [HumanMessage(content=ai_content)]

        async def _call():
            if hasattr(NodesGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await NodesGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, NodesGenerationAgent._STRUCTURED_MODEL.invoke, payload)

        _log_info("Calling LLM (structured)")
        res = await _retry_async(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:structured",
        )
        _log_info("LLM (structured) call succeeded")
        return res

    # ----- Inner graph nodes (async) -----

    @staticmethod
    async def _agent_node(state: _MongoNodesState):
        _log_tool_activity(state["messages"], ai_msg=None)
        ai = await NodesGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _MongoNodesState):
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
        # Final fallback: last message content
        if ai_content is None:
            ai_content = msgs[-1].content

        final_obj = await NodesGenerationAgent._invoke_structured(ai_content)
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoNodesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            _log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ----- RunnableLambda wrappers (avoid staticmethod/coroutine ambiguity) -----

    @staticmethod
    async def _agent_node_async(state: _MongoNodesState):
        return await NodesGenerationAgent._agent_node(state)

    @staticmethod
    async def _respond_node_async(state: _MongoNodesState):
        return await NodesGenerationAgent._respond_node(state)

    # ----- Compile inner graph -----

    @classmethod
    def _get_inner_graph(cls):
        if cls._compiled_graph is not None:
            return cls._compiled_graph

        g = StateGraph(_MongoNodesState)
        g.add_node("agent", RunnableLambda(cls._agent_node_async))
        g.add_node("respond", RunnableLambda(cls._respond_node_async))
        g.add_node("tools", ToolNode(cls.TOOLS, tags=["mongo-tools+arith-tools"]))
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", cls._should_continue, {"continue": "tools", "respond": "respond"})
        g.add_edge("tools", "agent")
        g.add_edge("respond", END)

        cls._compiled_graph = g.compile()
        return cls._compiled_graph

    # ----------------- Utilities -----------------

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
    async def _gen_once(per_topic_summary_json: str, thread_id: str, nodes_error: str) -> TopicWithNodesSchema:
        class AtTemplate(Template):
            delimiter = "@"

        tpl = AtTemplate(NODES_AGENT_PROMPT)
        sys_content = tpl.substitute(
            per_topic_summary_json=per_topic_summary_json,
            thread_id=thread_id,
            nodes_error=nodes_error,
        )

        sys_msg = SystemMessage(content=sys_content)
        trigger = HumanMessage(content="Based on the provided instructions please start the process")

        graph = NodesGenerationAgent._get_inner_graph()
        result = await graph.ainvoke({"messages": [sys_msg, trigger]})
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

        # Snapshot to detect mutation of discussion_summary_per_topic during generation
        snapshot = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True,
        )

        topic_with_nodes_list: List[TopicWithNodesSchema] = []

        for dspt_obj in zip(topics_list, summaries_list):
            per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)
            resp = await NodesGenerationAgent._gen_once(per_topic_summary_json, state.id, state.nodes_error)
            topic_with_nodes_list.append(resp)

        # Verify the source summaries were not mutated
        after = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True,
        )
        if after != snapshot:
            raise RuntimeError("discussion_summary_per_topic mutated during node generation")

        state.nodes = NodesSchema(
            topics_with_nodes=[t.model_dump() if hasattr(t, "model_dump") else t for t in topic_with_nodes_list]
        )

        # Gated final-output logging
        rendered = _render_nodes_result(state.nodes.model_dump_json(indent=2))
        _log_info(f"Nodes generation completed | output={rendered}")

        return state

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Validate container and per-topic schemas. Return True to regenerate on any error.
        """
        global _nodes_retry_counter

        if getattr(state, "nodes", None) is None:
            return True

        # 1) Validate container schema
        try:
            NodesSchema.model_validate(state.nodes.model_dump() if hasattr(state.nodes, "model_dump") else state.nodes)
        except ValidationError as ve:
            state.nodes_error += (
                "The previous generated o/p did not follow the given schema as it got following errors:\n"
                + (getattr(state, "nodes_error", "") or "")
                + "\n[NodesSchema ValidationError]\n"
                + str(ve)
                + "\n"
            )
            _log_retry(f"[NodesGen][ValidationError] Container NodesSchema invalid\n {ve}", _nodes_retry_counter)
            _nodes_retry_counter += 1
            return True

        # 2) Validate each TopicWithNodes
        try:
            topics_payload = (
                state.nodes.topics_with_nodes
                if hasattr(state.nodes, "topics_with_nodes")
                else state.nodes.get("topics_with_nodes", [])
            )
        except Exception as e:
            state.nodes_error += "\n[NodesSchema Payload Error]\n" + str(e) + "\n"
            _log_retry(f"[NodesGen][ValidationError] Could not read topics_with_nodes: {e}", _nodes_retry_counter)
            _nodes_retry_counter += 1
            return True

        any_invalid = False
        for idx, item in enumerate(topics_payload):
            try:
                TopicWithNodesSchema.model_validate(item.model_dump() if hasattr(item, "model_dump") else item)
            except ValidationError as ve:
                any_invalid = True
                state.nodes_error += f"\n[TopicWithNodesSchema ValidationError idx={idx}]\n{ve}\n"

        if any_invalid:
            _log_retry("node schema error", _nodes_retry_counter)
            _nodes_retry_counter += 1
        return any_invalid

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Graph for node generation:
        START -> nodes_generator -> (should_regenerate ? nodes_generator : END)
        """
        g = StateGraph(state_schema=AgentInternalState)
        g.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
        g.add_edge(START, "nodes_generator")
        g.add_conditional_edges(
            "nodes_generator",
            NodesGenerationAgent.should_regenerate,
            {True: "nodes_generator", False: END},
        )
        return g.compile(checkpointer=checkpointer, name="Nodes Generation Agent")


if __name__ == "__main__":
    graph = NodesGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
