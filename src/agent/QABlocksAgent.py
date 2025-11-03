# =============================================================================
# Module: qa_block_generation_agent
# =============================================================================
# Purpose
#   Generate structured discussion nodes for interview topics using a ReAct-style
#   agent architecture with concurrent processing, robust error handling, and
#   comprehensive logging. Each topic is processed independently via an inner LLM loop
#   with Mongo-backed tools, while the outer graph manages aggregation and validation.
#
# Architecture
#   Inner Graph (per topic):
#     agent ─► (tools)* ─► respond
#       - agent: LLM with tool access for planning and execution
#       - tools: ThreadPoolExecutor-backed tool calls with timeouts/retries
#       - respond: Coerce final tool-free response to QASetsSchema
#
#   Outer Graph:
#     START ──► qablock_generator
#               ├─► qablock_generator (regenerate if invalid)
#               └─► END                    (valid output)
#
# Key Features
#   • Concurrent topic processing via asyncio.gather
#   • Robust JSON extraction and schema coercion
#   • Exponential backoff retries for LLM/tools
#   • Thread-safe logging with optional per-thread files
#   • Graceful shutdown of thread pools and resources
#   • Comprehensive input/output validation
#   • Strict QA block validation:
#     - One block per deep-dive node
#     - Exactly 7 QA items per block
#     - No "Easy" counter questions allowed
#
# Environment Variables
#   Logging Configuration:
#     QA_AGENT_LOG_DIR        Base directory for log files
#     QA_AGENT_LOG_FILE       Log filename (default: qa_agent.log)
#     QA_AGENT_LOG_LEVEL      DEBUG|INFO|WARNING|ERROR|CRITICAL
#     QA_AGENT_LOG_ROTATE_WHEN      Rotation schedule (default: midnight)
#     QA_AGENT_LOG_ROTATE_INTERVAL  Rotation interval (default: 1)
#     QA_AGENT_LOG_BACKUP_COUNT    Maximum backups (default: 365)
#
#   LLM Settings:
#     QA_AGENT_LLM_TIMEOUT_SECONDS         Timeout per LLM call (default: 120)
#     QA_AGENT_LLM_RETRIES                 Maximum retries (default: 2)
#     QA_AGENT_LLM_RETRY_BACKOFF_SECONDS   Base backoff time (default: 2.5)
#
#   Tool Settings:
#     QA_AGENT_TOOL_TIMEOUT_SECONDS        Timeout per tool call (default: 30)
#     QA_AGENT_TOOL_RETRIES               Maximum retries (default: 2)
#     QA_AGENT_TOOL_RETRY_BACKOFF_SECONDS  Base backoff time (default: 1.5)
#     QA_AGENT_TOOL_MAX_WORKERS           ThreadPool size (default: 8)
#
#   Logging Controls:
#     QA_AGENT_TOOL_LOG_PAYLOAD           Tool payload logging (off|summary|full)
#     QA_AGENT_RESULT_LOG_PAYLOAD         Result payload logging (off|summary|full)
#     QA_AGENT_LOG_SPLIT_BY_THREAD        Enable per-thread log files (0|1)
#
# Dependencies
#   • langchain-core: LLM interfaces and tools
#   • langgraph: Graph orchestration
#   • pydantic: Schema validation
#   • pymongo: MongoDB operations
# =============================================================================

import asyncio
import atexit
import contextlib
import copy
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from string import Template
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import PrivateAttr, ValidationError

from src.model_handling import llm_qa as _llm_client
from src.mongo_tools import get_mongo_tools
from src.prompt.qa_agent_prompt import QA_BLOCK_AGENT_PROMPT
from src.schema.agent_schema import AgentInternalState
from src.schema.output_schema import QASetsSchema

# ==============================
# Configuration
# ==============================


@dataclass(frozen=True)
class QAConfig:
    agent_name: str = os.getenv("QA_AGENT_NAME", "qa_block_generation_agent")

    log_dir: str = os.getenv("QA_AGENT_LOG_DIR", "logs")
    log_file: str = os.getenv("QA_AGENT_LOG_FILE", f"{agent_name}.log")
    log_level: int = getattr(
        logging, os.getenv("QA_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    log_rotate_when: str = os.getenv("QA_AGENT_LOG_ROTATE_WHEN", "midnight")
    log_rotate_interval: int = int(os.getenv("QA_AGENT_LOG_ROTATE_INTERVAL", "1"))
    log_backup_count: int = int(os.getenv("QA_AGENT_LOG_BACKUP_COUNT", "365"))

    llm_timeout_s: float = float(os.getenv("QA_AGENT_LLM_TIMEOUT_SECONDS", "120"))
    llm_retries: int = int(os.getenv("QA_AGENT_LLM_RETRIES", "2"))
    llm_backoff_base_s: float = float(
        os.getenv("QA_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5")
    )

    tool_timeout_s: float = float(os.getenv("QA_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
    tool_retries: int = int(os.getenv("QA_AGENT_TOOL_RETRIES", "2"))
    tool_backoff_base_s: float = float(
        os.getenv("QA_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5")
    )
    tool_max_workers: int = int(os.getenv("QA_AGENT_TOOL_MAX_WORKERS", "8"))

    tool_log_payload: str = (
        os.getenv("QA_AGENT_TOOL_LOG_PAYLOAD", "off").strip().lower()
    )
    result_log_payload: str = (
        os.getenv("QA_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()
    )
    # Split logs by thread id (default on). Set to 0/false to keep one shared file.
    split_log_by_thread: bool = os.getenv(
        "QA_AGENT_LOG_SPLIT_BY_THREAD", "1"
    ).strip().lower() in ("1", "true", "yes", "y")


CFG = QAConfig()
_EXECUTOR = ThreadPoolExecutor(max_workers=CFG.tool_max_workers)

# Global retry counter for logging only
qa_retry_counter = 1

# Current thread id for logging context
THREAD_ID_VAR: ContextVar[str] = ContextVar("thread_id", default="-")


class _ThreadIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.thread_id = THREAD_ID_VAR.get()
        except Exception:
            record.thread_id = "-"
        return True


# Avoid duplicate handlers per thread id
_THREAD_FILE_HANDLERS: Dict[str, logging.Handler] = {}


def shutdown_executor():
    # Graceful shutdown; also cancels pending futures if supported by runtime.
    try:
        _EXECUTOR.shutdown(wait=True, cancel_futures=True)
    except TypeError:
        # Python <3.9 compatibility (no cancel_futures)
        with contextlib.suppress(Exception):
            _EXECUTOR.shutdown(wait=True)
    except Exception:
        pass


def _classify_provider_error(exc: Exception) -> tuple[str, Dict[str, Any]]:
    """
    Return (reason, extra) where reason is one of:
      'billing/quota' | 'auth' | 'permission' | 'rate_limited'
      | 'http_<code>' | 'timeout' | 'provider_api_error' | 'unknown'
    and extra contains structured hints (status_code, provider_error/code, etc.).
    """
    reason = "unknown"
    extra: Dict[str, Any] = {"error": str(exc)}

    # asyncio timeout from wait_for(...)
    import asyncio as _asyncio

    if isinstance(exc, _asyncio.TimeoutError):
        return "timeout", extra

    # httpx (common under LangChain / SDKs)
    try:
        import httpx  # type: ignore
    except Exception:
        httpx = None

    if httpx and isinstance(exc, httpx.HTTPStatusError):
        resp = exc.response
        extra["status_code"] = resp.status_code
        try:
            body = resp.json()
            extra["provider_error"] = body
            err = (body.get("error") or {}) if isinstance(body, dict) else {}
            code = err.get("code") or err.get("type")
            if code:
                extra["provider_error_code"] = code
                if code in {"insufficient_quota", "billing_hard_limit_reached"}:
                    reason = "billing/quota"
                elif code in {"invalid_api_key", "authentication_error"}:
                    reason = "auth"
        except Exception:
            pass
        if reason == "unknown":
            if resp.status_code == 401:
                reason = "auth"
            elif resp.status_code == 403:
                reason = "permission"
            elif resp.status_code == 429:
                reason = "rate_limited"
            else:
                reason = f"http_{resp.status_code}"
        return reason, extra

    # OpenAI (if you ever call SDK directly beneath LangChain)
    try:
        import openai  # type: ignore

        if isinstance(exc, openai.RateLimitError):
            return "rate_limited", extra
        if isinstance(exc, openai.AuthenticationError):
            return "auth", extra
        if isinstance(exc, openai.PermissionDeniedError):
            return "permission", extra
        if isinstance(exc, openai.APIError):
            return "provider_api_error", extra
    except Exception:
        pass

    return reason, extra


def _attach_thread_file_handler(thread_id: str) -> None:
    """Attach a per-thread TimedRotatingFileHandler writing to <log_dir>/<thread_id>/<agent>.log."""
    if not CFG.split_log_by_thread or not thread_id:
        return
    if thread_id in _THREAD_FILE_HANDLERS:
        return

    # Sanitize and build per-thread folder: <log_dir>/<thread_id>/
    safe_tid = re.sub(r"[^\w.-]", "_", str(thread_id))  # requires: import re
    thread_dir = os.path.join(CFG.log_dir, safe_tid)
    os.makedirs(thread_dir, exist_ok=True)

    # File name stays constant: <agent>.log (CFG.log_file)
    path = os.path.join(thread_dir, CFG.log_file)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | tid=%(thread_id)s | %(message)s"
    )
    handler = TimedRotatingFileHandler(
        path,
        when=CFG.log_rotate_when,
        interval=CFG.log_rotate_interval,
        backupCount=CFG.log_backup_count,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    logging.raiseExceptions = False
    handler.setLevel(CFG.log_level)
    handler.setFormatter(fmt)
    handler.set_name(f"file::thread::{safe_tid}")
    handler.addFilter(_ThreadIdFilter())

    LOGGER.addHandler(handler)
    _THREAD_FILE_HANDLERS[thread_id] = handler


def _detach_thread_file_handler(thread_id: str) -> None:
    """Close and remove the per-thread file handler if present (graceful cleanup only)."""
    h = _THREAD_FILE_HANDLERS.pop(thread_id, None)
    if h:
        try:
            LOGGER.removeHandler(h)
        finally:
            with contextlib.suppress(Exception):
                h.close()


def _close_all_thread_file_handlers() -> None:
    """Best-effort close of all per-thread file handlers."""
    for tid in list(_THREAD_FILE_HANDLERS.keys()):
        _detach_thread_file_handler(tid)


# ==============================
# Logging
# ==============================


def _get_logger() -> logging.Logger:
    """Configure and return the agent logger.

    Adds a stdout handler and, if `CFG.split_log_by_thread` is False, a
    timed-rotating file handler. Injects `thread_id` via `_ThreadIdFilter`.
    Idempotent: if handlers exist, returns the existing logger.
    """
    logger = logging.getLogger(CFG.agent_name)
    if logger.handlers:
        return logger

    logger.setLevel(CFG.log_level)
    logger.propagate = False

    os.makedirs(CFG.log_dir, exist_ok=True)
    path = os.path.join(CFG.log_dir, CFG.log_file)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | tid=%(thread_id)s | %(message)s"
    )
    tid_filter = _ThreadIdFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(CFG.log_level)
    console.setFormatter(fmt)
    console.addFilter(tid_filter)
    logger.addHandler(console)

    if not CFG.split_log_by_thread:
        rotate_file = TimedRotatingFileHandler(
            path,
            when=CFG.log_rotate_when,
            interval=CFG.log_rotate_interval,
            backupCount=CFG.log_backup_count,
            encoding="utf-8",
            utc=False,
            delay=True,
        )
        logging.raiseExceptions = False
        rotate_file.setLevel(CFG.log_level)
        rotate_file.setFormatter(fmt)
        rotate_file.set_name("file::shared")
        rotate_file.addFilter(tid_filter)
        logger.addHandler(rotate_file)

    return logger


def with_thread_context(fn):
    """Sets THREAD_ID_VAR from state.id, attaches per-thread handler, restores afterwards."""

    @wraps(fn)
    async def _inner(*args, **kwargs):
        state = args[0] if args else kwargs.get("state")
        tid = getattr(state, "id", None) or "-"
        token = THREAD_ID_VAR.set(tid)
        try:
            _attach_thread_file_handler(tid)
            return await fn(*args, **kwargs)
        finally:
            THREAD_ID_VAR.reset(token)

    return _inner


LOGGER = _get_logger()


def log_info(msg: str) -> None:
    LOGGER.info(msg)


def log_warning(msg: str) -> None:
    LOGGER.warning(msg)


# ==============================
# JSON + compact logging helpers
# ==============================


def _looks_like_json(text: str) -> bool:
    t = (text or "").strip()
    return (t.startswith("{") and t.endswith("}")) or (
        t.startswith("[") and t.endswith("]")
    )


def _jsonish(v: Any) -> Any:
    if isinstance(v, str) and _looks_like_json(v):
        try:
            return json.loads(v)
        except Exception:
            return v
    if isinstance(v, dict):
        return {k: _jsonish(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonish(x) for x in v]
    return v


def _compact(v: Any) -> str:
    try:
        return (
            json.dumps(v, ensure_ascii=False, indent=2)
            if isinstance(v, (dict, list))
            else str(v)
        )
    except Exception:
        return str(v)


def _pydantic_obj(o: Any) -> Any:
    # pydantic v2
    if hasattr(o, "model_dump_json"):
        try:
            return json.loads(o.model_dump_json())
        except Exception:
            pass
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            pass
    # pydantic v1 / dict-ish
    if hasattr(o, "dict"):
        try:
            return o.dict()
        except Exception:
            pass
    return o


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


def _gated_tool_payload(payload: Any) -> str:
    m = CFG.tool_log_payload
    if m == "off":
        return "<hidden>"
    if m == "summary":
        return _summarize_tool_payload(payload)
    return _compact(_jsonish(payload))


def _summarize_qa_result(payload: Any) -> str:
    """
    Compact view of QASetsSchema:
      • num topics
      • for first few topics: topic name, blocks count, per-block qa_items count
    """
    try:
        data = _pydantic_obj(payload)
        qa_sets = (
            data.get("qa_sets")
            if isinstance(data, dict)
            else getattr(payload, "qa_sets", None)
        )

        if isinstance(qa_sets, list):
            n = len(qa_sets)
            details: List[Dict[str, Any]] = []
            for qs in qa_sets[:4]:
                d = _pydantic_obj(qs) if not isinstance(qs, dict) else qs
                topic = d.get("topic") or "Unknown"
                blocks = d.get("qa_blocks") or []
                counts: List[int] = []
                for b in blocks[:6]:
                    bd = _pydantic_obj(b) if not isinstance(b, dict) else b
                    items = bd.get("qa_items") or []
                    counts.append(len(items))
                details.append(
                    {
                        "topic": topic,
                        "blocks": len(blocks),
                        "qa_items_first_blocks": counts,
                    }
                )
            return f"topics={n} details={details}{'...' if n > 4 else ''}"

        if isinstance(data, dict):
            return f"dict(keys={list(data.keys())[:8]})"
        if isinstance(data, list):
            return f"list(len={len(data)})"
        return type(data).__name__
    except Exception:
        return "<unavailable>"


def _gated_qa_result(payload: Any) -> str:
    m = CFG.result_log_payload
    if m == "off":
        return "<hidden>"
    if m == "summary":
        return _summarize_qa_result(payload)
    return _compact(_pydantic_obj(payload))


def log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    if not messages:
        return

    planned = getattr(ai_msg, "tool_calls", None)
    if planned:
        log_info("Tool plan:")
        for tc in planned:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            LOGGER.info(f"  planned -> {name} args={_gated_tool_payload(args)}")

    tool_msgs: List[Any] = []
    i = len(messages) - 1
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        tool_msgs.append(messages[i])
        i -= 1
    if not tool_msgs:
        return

    log_info("Tool results:")
    for tm in tool_msgs:
        content = getattr(tm, "content", None)
        LOGGER.info(
            f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_gated_tool_payload(content)}"
        )


def log_retry_iteration(
    reason: str, iteration: int, extra: Optional[dict] = None
) -> None:
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


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
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= retries:
        try:
            return await asyncio.wait_for(op_factory(), timeout=timeout_s)
        except Exception as exc:
            last_exc = exc
            log_retry_iteration(
                retry_reason, iteration_start + attempt, {"error": str(exc)}
            )
            attempt += 1
            if attempt > retries:
                break
            await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))

    assert last_exc is not None
    # One terminal line with reason + details (thread id is already injected)
    try:
        reason, extra = _classify_provider_error(last_exc)
    except Exception:
        reason, extra = "unknown", {"error": str(last_exc)}

    LOGGER.error(
        "Terminal failure after %d attempt(s) | context=%s | reason=%s | extra=%s",
        retries + 1,
        retry_reason,
        reason,
        extra,
    )
    raise last_exc


# ==============================
# Tool wrapper (timeout + retry)
# ==============================


class RetryTool(BaseTool):
    """
    Wrap BaseTool to add timeout + retries (sync/async). Preserves metadata for
    bind_tools / ToolNode compatibility.
    """

    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff: float = PrivateAttr()

    def __init__(
        self, inner: BaseTool, *, retries: int, timeout_s: float, backoff_base_s: float
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
            fut = _EXECUTOR.submit(_call_once)
            try:
                return fut.result(timeout=self._timeout_s)
            except FuturesTimeout as exc:
                # Ensure we don't keep a pending future around
                with contextlib.suppress(Exception):
                    fut.cancel()
                last_exc = exc
                log_retry_iteration(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    fut.cancel()
                last_exc = exc
                log_retry_iteration(
                    f"tool_error:{self.name}", attempt + 1, {"error": str(exc)}
                )
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)

        async def _call_once():
            if hasattr(self._inner, "_arun"):
                return await getattr(self._inner, "_arun")(
                    *args, **{**kwargs, "config": config}
                )
            loop = asyncio.get_running_loop()
            # Use the shared executor so we control lifecycle
            return await loop.run_in_executor(
                _EXECUTOR,
                lambda: self._inner._run(*args, **{**kwargs, "config": config}),
            )

        return await _retry_async(
            _call_once,
            retries=self._retries,
            timeout_s=self._timeout_s,
            backoff_base_s=self._backoff,
            retry_reason=f"tool_async:{self.name}",
        )


# ==============================
# Inner ReAct state (per-topic run)
# ==============================


class _QAInnerState(MessagesState):
    """State container for the inner ReAct loop that generates QA blocks."""

    final_response: QASetsSchema


class QABlockGenerationAgent:
    """
    Generate QA blocks for each topic's deep-dive nodes using a tool-enabled
    inner loop, coerce to QASetsSchema, and apply strict post-generation checks.
    """

    llm = _llm_client

    # Tools (wrapped with retry/timeout)
    _RAW_TOOLS: List[BaseTool] = get_mongo_tools(llm=llm)
    MONGO_TOOLS: List[BaseTool] = [
        RetryTool(
            t,
            retries=CFG.tool_retries,
            timeout_s=CFG.tool_timeout_s,
            backoff_base_s=CFG.tool_backoff_base_s,
        )
        for t in _RAW_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(QASetsSchema)

    _compiled_inner_graph = None  # cache

    # ----- LLM invokers -----

    @staticmethod
    async def _invoke_agent(messages: Sequence[Any]) -> Any:
        async def _call():
            if hasattr(QABlockGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await QABlockGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _EXECUTOR, QABlockGenerationAgent._AGENT_MODEL.invoke, messages
            )

        log_info("Calling LLM (agent)")
        res = await _retry_async(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:agent",
        )
        log_info("LLM (agent) call succeeded")
        return res

    @staticmethod
    async def _invoke_structured(ai_content: str) -> QASetsSchema:
        payload = [HumanMessage(content=ai_content)]

        async def _call():
            if hasattr(QABlockGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await QABlockGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _EXECUTOR, QABlockGenerationAgent._STRUCTURED_MODEL.invoke, payload
            )

        log_info("Calling LLM (structured)")
        res = await _retry_async(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:structured",
        )
        log_info("LLM (structured) call succeeded")
        return res

    # ----- Inner graph nodes (async) -----

    @staticmethod
    async def _agent_node(state: _QAInnerState):
        log_tool_activity(messages=state["messages"], ai_msg=None)
        ai = await QABlockGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _QAInnerState):
        msgs = state["messages"]
        ai_content: Optional[str] = None

        # Prefer last assistant msg without tool calls
        for m in reversed(msgs):
            if getattr(m, "type", None) in ("ai", "assistant") and not getattr(
                m, "tool_calls", None
            ):
                ai_content = m.content
                break
        # Fallback: any assistant
        if ai_content is None:
            for m in reversed(msgs):
                if getattr(m, "type", None) in ("ai", "assistant"):
                    ai_content = m.content
                    break
        # Final fallback
        if ai_content is None:
            ai_content = msgs[-1].content

        final_obj = await QABlockGenerationAgent._invoke_structured(ai_content)
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _QAInnerState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(messages=state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ----- Compile inner graph -----

    @classmethod
    def _get_inner_graph(cls):
        if cls._compiled_inner_graph is not None:
            return cls._compiled_inner_graph

        g = StateGraph(_QAInnerState)
        g.add_node("agent", RunnableLambda(cls._agent_node))
        g.add_node("respond", RunnableLambda(cls._respond_node))
        g.add_node("tools", ToolNode(cls.MONGO_TOOLS, tags=["mongo-tools"]))
        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent", cls._should_continue, {"continue": "tools", "respond": "respond"}
        )
        g.add_edge("tools", "agent")
        g.add_edge("respond", END)

        cls._compiled_inner_graph = g.compile()
        return cls._compiled_inner_graph

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
        d = QABlockGenerationAgent._as_dict(obj)
        for k in ("topic", "name", "title", "label"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    @staticmethod
    def _canon(text: str) -> str:
        t = (text or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^\w\s]", "", t)
        return t

    @staticmethod
    def _topic_from_summary(summary_obj: Any) -> str:
        d = QABlockGenerationAgent._as_dict(summary_obj)
        for k in ("topic", "name", "title", "label", "discussion_topic", "heading"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    @staticmethod
    async def _gen_for_topic(
        topic_name: str,
        discussion_summary_json: str,
        deep_dive_nodes_json: str,
        thread_id: str,
        qa_error: str = "",
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate QA blocks for a single topic and enforce:
        - number of blocks == number of deep-dive nodes
        - each block has exactly 7 qa_items
        - 'Counter Question' items are never 'Easy'
        Returns (qa_set_dict, error_message). error_message == "" means OK.
        """
        token = THREAD_ID_VAR.set(thread_id or "-")
        try:
            _attach_thread_file_handler(thread_id or "-")

            # --- Parse deep-dive nodes safely ---
            parse_err = ""
            try:
                deep_dive_nodes = json.loads(deep_dive_nodes_json or "[]")
                if not isinstance(deep_dive_nodes, list):
                    raise ValueError("deep_dive_nodes_json is not a JSON array")
            except Exception as e:
                deep_dive_nodes = []
                parse_err = f"Deep-dive nodes JSON parse error: {e}"
            n_blocks = len(deep_dive_nodes)

            # --- Build prompt with explicit count directive ---
            class AtTemplate(Template):
                delimiter = "@"

            count_directive = (
                f"\nIMPORTANT COUNT RULE:\n"
                f"- This topic has {n_blocks} deep-dive nodes. "
                f"Output exactly {n_blocks} QA blocks (one per deep-dive node, in order).\n"
            )

            tpl = AtTemplate(QA_BLOCK_AGENT_PROMPT + count_directive)
            sys_content = tpl.substitute(
                discussion_summary=discussion_summary_json,
                deep_dive_nodes=deep_dive_nodes_json,
                thread_id=thread_id,
                qa_error=qa_error or "",
            )

            sys_msg = SystemMessage(content=sys_content)
            trigger = HumanMessage(
                content="Based on the provided instructions please start the process"
            )

            # --- Invoke inner graph ---
            try:
                graph = QABlockGenerationAgent._get_inner_graph()
                result = await graph.ainvoke({"messages": [sys_msg, trigger]})
                schema = (
                    result["final_response"] if isinstance(result, dict) else result
                )  # QASetsSchema or dict
            except (ValidationError, OutputParserException) as e:
                return {
                    "topic": topic_name,
                    "qa_blocks": [],
                }, f"Parser/Schema error: {e}"
            except Exception as e:
                return {"topic": topic_name, "qa_blocks": []}, f"Generation error: {e}"

            # --- Normalize to plain dict ---
            obj = (
                schema.model_dump()
                if hasattr(schema, "model_dump")
                else (schema if isinstance(schema, dict) else {})
            )
            qa_sets = obj.get("qa_sets") or []
            if not isinstance(qa_sets, list) or not qa_sets:
                err = "No qa_sets produced."
                if parse_err:
                    err = f"{err} {parse_err}"
                return {"topic": topic_name, "qa_blocks": []}, err

            # We expect one QA set per topic; take the first and enforce rules
            one = qa_sets[0] if isinstance(qa_sets[0], dict) else {}
            one["topic"] = topic_name
            blocks = one.get("qa_blocks") or []
            if not isinstance(blocks, list):
                blocks = []

            errs: List[str] = []
            if parse_err:
                errs.append(parse_err)
            if len(blocks) != n_blocks:
                errs.append(f"Expected {n_blocks} blocks, got {len(blocks)}.")

            for i, b in enumerate(blocks, start=1):
                bd = b if isinstance(b, dict) else {}
                qi = bd.get("qa_items") or []
                if not isinstance(qi, list):
                    qi = []
                if len(qi) != 7:
                    errs.append(f"Block {i} must have 7 qa_items, got {len(qi)}.")
                for item in qi:
                    it = item if isinstance(item, dict) else {}
                    if (
                        it.get("q_type") == "Counter Question"
                        and it.get("q_difficulty") == "Easy"
                    ):
                        errs.append(
                            f"Block {i} has an Easy counter (qa_id={it.get('qa_id')}); not allowed."
                        )

            if errs:
                return one, " ; ".join(errs)

            return one, ""

        finally:
            THREAD_ID_VAR.reset(token)

    # ----------------- Main outer node -----------------

    @staticmethod
    @with_thread_context
    async def qablock_generator(state: AgentInternalState) -> AgentInternalState:
        """
        For each topic in state.nodes.topics_with_nodes:
          • locate its discussion summary
          • collect its deep-dive nodes
          • run inner loop to produce QA blocks
        Aggregate into state.qa_blocks (QASetsSchema) or record errors.
        """
        if state.interview_topics is None or not getattr(
            state.interview_topics, "interview_topics", None
        ):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")
        if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
            raise ValueError(
                "nodes (topics_with_nodes) are required before QA block generation."
            )

        # Normalize summaries list
        raw = state.discussion_summary_per_topic
        if hasattr(raw, "discussion_topics"):
            summaries_list = list(raw.discussion_topics)
        elif isinstance(raw, (list, tuple)):
            summaries_list = list(raw)
        elif isinstance(raw, dict) and "discussion_topics" in raw:
            summaries_list = list(raw["discussion_topics"])
        else:
            raise ValueError(
                "discussion_summary_per_topic has no 'discussion_topics' field"
            )

        # Quick index by canonical topic
        summaries_by_can: Dict[str, Any] = {}
        for s in summaries_list:
            nm = QABlockGenerationAgent._topic_from_summary(s)
            summaries_by_can[QABlockGenerationAgent._canon(nm)] = s

        final_sets: List[Dict[str, Any]] = []
        accumulated_errs: List[str] = []
        covered: set[str] = set()

        # Deep-dive aliases
        DEEP_DIVE_ALIASES = {
            "deep dive",
            "deep_dive",
            "deep-dive",
            "probe",
            "follow up",
            "follow-up",
        }

        log_info("QA block generation started")

        for topic_entry in state.nodes.topics_with_nodes:
            topic_dict = (
                topic_entry.model_dump()
                if hasattr(topic_entry, "model_dump")
                else dict(topic_entry)
            )
            topic_name = (
                topic_dict.get("topic")
                or QABlockGenerationAgent._get_topic_name(topic_entry)
                or "Unknown"
            )
            ckey = QABlockGenerationAgent._canon(topic_name)

            summary_obj = summaries_by_can.get(ckey)
            if summary_obj is None:
                accumulated_errs.append(
                    f"[QABlocks] No summary for '{topic_name}'; skipping topic this round."
                )
                covered.add(ckey)
                continue

            # Collect deep-dive nodes in order
            deep_dive_nodes: List[dict] = []
            for node in topic_dict.get("nodes") or []:
                qtype = str(node.get("question_type", "")).strip().lower()
                if qtype in DEEP_DIVE_ALIASES:
                    deep_dive_nodes.append(node)

            if not deep_dive_nodes:
                accumulated_errs.append(
                    f"[QABlocks] Topic '{topic_name}' has no deep-dive nodes; skipping this round."
                )
                covered.add(ckey)
                continue

            summary_json = json.dumps(
                summary_obj.model_dump()
                if hasattr(summary_obj, "model_dump")
                else summary_obj
            )
            deep_dive_nodes_json = json.dumps(copy.deepcopy(deep_dive_nodes))

            one_set, err = await QABlockGenerationAgent._gen_for_topic(
                topic_name=topic_name,
                discussion_summary_json=summary_json,
                deep_dive_nodes_json=deep_dive_nodes_json,
                thread_id=state.id,
                qa_error=getattr(state, "qa_error", "") or "",
            )

            if err:
                accumulated_errs.append(f"[{topic_name}] {err}")
            blocks = one_set.get("qa_blocks") or []
            if blocks:
                final_sets.append(one_set)
            else:
                accumulated_errs.append(
                    f"[{topic_name}] model returned 0 QA blocks; will retry."
                )
            covered.add(ckey)

        if final_sets:
            state.qa_blocks = QASetsSchema(qa_sets=final_sets)
        else:
            state.qa_blocks = None
            log_warning("QA block generation produced no valid blocks this pass")
            if not accumulated_errs:
                accumulated_errs.append(
                    "[QABlocks] No topics produced QA blocks this attempt."
                )

        if accumulated_errs:
            prev = getattr(state, "qa_error", "") or ""
            state.qa_error = (
                prev + ("\n" if prev else "") + "\n".join(accumulated_errs)
            ).strip()

        rendered = _gated_qa_result(state.qa_blocks.model_dump_json(indent=2))
        log_info(f"QA block generation before retry checks | output={rendered}")

        return state

    @staticmethod
    @with_thread_context
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Regenerate when:
          • qa_blocks is None (no valid blocks produced yet)
          • schema validation fails
          • any topic ends with 0 blocks after validation
        """
        global qa_retry_counter

        if getattr(state, "qa_blocks", None) is None:
            log_retry_iteration(
                "qa_blocks is None (no valid blocks yet); retrying", qa_retry_counter
            )
            qa_retry_counter += 1
            return True

        # Validate container
        try:
            QASetsSchema.model_validate(
                state.qa_blocks.model_dump()
                if hasattr(state.qa_blocks, "model_dump")
                else state.qa_blocks
            )
        except ValidationError as ve:
            state.qa_error = (
                (getattr(state, "qa_error", "") or "")
                + ("\n" if getattr(state, "qa_error", "") else "")
                + "The previous generated o/p did not follow the given schema as it got following errors:\n"
                "[QABlockGen ValidationError]\n "
                f"{ve}"
            )
            log_retry_iteration(
                "Schema validation failed", qa_retry_counter, {"error": str(ve)}
            )
            qa_retry_counter += 1
            return True

        # Ensure each set has at least one block
        try:
            sets = (
                state.qa_blocks.qa_sets
                if hasattr(state.qa_blocks, "qa_sets")
                else state.qa_blocks.get("qa_sets", [])
            )
            if any(
                not (qs.get("qa_blocks") if isinstance(qs, dict) else qs.qa_blocks)
                for qs in sets
            ):
                log_retry_iteration(
                    "At least one topic has 0 qa_blocks after validation; retrying",
                    qa_retry_counter,
                )
                qa_retry_counter += 1
                return True
        except Exception as e:
            log_retry_iteration(
                "Introspection failed while checking qa_sets; allowing retry",
                qa_retry_counter,
                {"error": str(e)},
            )
            qa_retry_counter += 1
            return True

        rendered = _gated_qa_result(state.qa_blocks.model_dump_json(indent=2))
        log_info(f"QA block generation successfully completed | output={rendered}")
        qa_retry_counter = 1
        shutdown_executor()
        return False

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Graph for QA block generation:
        START -> qablock_generator -> (should_regenerate ? qablock_generator : END)
        """
        g = StateGraph(state_schema=AgentInternalState)
        g.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
        g.add_edge(START, "qablock_generator")
        g.add_conditional_edges(
            "qablock_generator",
            QABlockGenerationAgent.should_regenerate,
            {True: "qablock_generator", False: END},
        )
        return g.compile(checkpointer=checkpointer, name="QA Block Generation Agent")


# -------- Process-exit safety nets (no behavior change to main flow) --------
@atexit.register
def _shutdown_qa_agent_at_exit() -> None:
    with contextlib.suppress(Exception):
        _close_all_thread_file_handlers()
    with contextlib.suppress(Exception):
        shutdown_executor()


if __name__ == "__main__":
    graph = QABlockGenerationAgent.get_graph()
