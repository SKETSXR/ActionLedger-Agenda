# =============================================================================
# Module: topic_generation_agent
# =============================================================================
# Purpose
#   Generate structured interview topics from a generated summary using a ReAct-style
#   agent architecture with concurrent processing, robust error handling, and
#   comprehensive logging. Each topic generation attempt is processed via an inner LLM loop
#   with Mongo-backed tools, while the outer graph manages validation and regeneration.
#
# Architecture
#   Inner Graph:
#     agent ─► (tools)* ─► respond
#       - agent: LLM with tool access for planning and execution
#       - tools: ThreadPoolExecutor-backed tool calls with timeouts/retries
#       - respond: Coerce final tool-free response to CollectiveInterviewTopicSchema
#
#   Outer Graph:
#     START ──► topic_generator
#               ├─► topic_generator (regenerate if invalid)
#               └─► END                    (valid output)
#
# Key Features
#   • System prompt generation from prior outputs
#   • Robust JSON extraction and schema coercion
#   • Exponential backoff retries for LLM/tools
#   • Thread-safe logging with optional per-thread files
#   • Graceful shutdown of thread pools and resources
#   • Comprehensive input/output validation
#   • Strict topic validation:
#     - Total questions match summary
#     - Focus skills are valid leaf nodes
#     - All must-have skills are covered
#
# Environment Variables
#   Logging Configuration:
#     TOPIC_AGENT_LOG_DIR        Base directory for log files
#     TOPIC_AGENT_LOG_FILE       Log filename (default: topic_agent.log)
#     TOPIC_AGENT_LOG_LEVEL      DEBUG|INFO|WARNING|ERROR|CRITICAL
#     TOPIC_AGENT_LOG_ROTATE_WHEN      Rotation schedule (default: midnight)
#     TOPIC_AGENT_LOG_ROTATE_INTERVAL  Rotation interval (default: 1)
#     TOPIC_AGENT_LOG_BACKUP_COUNT    Maximum backups (default: 365)
#
#   LLM Settings:
#     TOPIC_AGENT_LLM_TIMEOUT_SECONDS         Timeout per LLM call (default: 90)
#     TOPIC_AGENT_LLM_RETRIES                 Maximum retries (default: 2)
#     TOPIC_AGENT_LLM_RETRY_BACKOFF_SECONDS   Base backoff time (default: 2.5)
#
#   Tool Settings:
#     TOPIC_AGENT_TOOL_TIMEOUT_SECONDS        Timeout per tool call (default: 30)
#     TOPIC_AGENT_TOOL_RETRIES               Maximum retries (default: 2)
#     TOPIC_AGENT_TOOL_RETRY_BACKOFF_SECONDS  Base backoff time (default: 1.5)
#     TOPIC_AGENT_TOOL_MAX_WORKERS           ThreadPool size (default: 8)
#
#   Logging Controls:
#     TOPIC_AGENT_TOOL_LOG_PAYLOAD           Tool payload logging (off|summary|full)
#     TOPIC_AGENT_RESULT_LOG_PAYLOAD         Result payload logging (off|summary|full)
#     TOPIC_AGENT_LOG_SPLIT_BY_THREAD        Enable per-thread log files (0|1)
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
from typing import Any, Callable, Coroutine, Optional, Sequence, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import PrivateAttr

from src.model_handling import llm_tg as _llm_client
from src.mongo_tools import get_mongo_tools
from src.prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from src.schema.agent_schema import AgentInternalState
from src.schema.output_schema import CollectiveInterviewTopicSchema

# ==============================
# Configuration
# ==============================

AGENT_NAME = "topic_generation_agent"

FEEDBACK_HEADER = (
    "<Please don't miss/skip on any of these skills from the provided "
    "MUST_SKILLS set in your focus areas of the last topic of General Skill Assessment>\n"
)


@dataclass(frozen=True)
class TopicAgentConfig:
    log_dir: str = os.getenv("TOPIC_AGENT_LOG_DIR", "logs")
    log_file: str = os.getenv("TOPIC_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
    log_level: int = getattr(
        logging, os.getenv("TOPIC_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    log_rotate_when: str = os.getenv("TOPIC_AGENT_LOG_ROTATE_WHEN", "midnight")
    log_rotate_interval: int = int(os.getenv("TOPIC_AGENT_LOG_ROTATE_INTERVAL", "1"))
    log_backup_count: int = int(os.getenv("TOPIC_AGENT_LOG_BACKUP_COUNT", "365"))

    llm_timeout_s: float = float(os.getenv("TOPIC_AGENT_LLM_TIMEOUT_SECONDS", "90"))
    llm_retries: int = int(os.getenv("TOPIC_AGENT_LLM_RETRIES", "2"))
    llm_backoff_base_s: float = float(
        os.getenv("TOPIC_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5")
    )

    tool_timeout_s: float = float(os.getenv("TOPIC_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
    tool_retries: int = int(os.getenv("TOPIC_AGENT_TOOL_RETRIES", "2"))
    tool_backoff_base_s: float = float(
        os.getenv("TOPIC_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5")
    )
    tool_max_workers: int = int(os.getenv("TOPIC_AGENT_TOOL_MAX_WORKERS", "8"))

    tool_log_payload: str = (
        os.getenv("TOPIC_AGENT_TOOL_LOG_PAYLOAD", "off").strip().lower()
    )
    result_log_payload: str = (
        os.getenv("TOPIC_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()
    )
    # Split logs by thread id (default on). Set to 0/false for a single shared file.
    split_log_by_thread: bool = os.getenv(
        "TOPIC_AGENT_LOG_SPLIT_BY_THREAD", "1"
    ).strip().lower() in ("1", "true", "yes", "y")
    # Feedback logging toggle
    log_feedbacks: bool = os.getenv(
        "TOPIC_AGENT_LOG_FEEDBACKS", "0"
    ).strip().lower() in ("1", "true", "yes", "y")

    feedback_log_level: str = (
        os.getenv("TOPIC_AGENT_FEEDBACK_LOG_LEVEL", "INFO").strip().upper()
    )


CFG = TopicAgentConfig()

# Single shared executor for sync tools with timeouts
_EXECUTOR = ThreadPoolExecutor(max_workers=CFG.tool_max_workers)

# Thread id for logging context
THREAD_ID_VAR: ContextVar[str] = ContextVar("thread_id", default="-")


class _ThreadIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.thread_id = THREAD_ID_VAR.get()
        except Exception:
            record.thread_id = "-"
        return True


# Avoid duplicate per-thread handlers
_THREAD_FILE_HANDLERS: dict[str, logging.Handler] = {}


def shutdown_executor() -> None:
    # Graceful shutdown; cancel any pending futures when supported.
    try:
        _EXECUTOR.shutdown(wait=True, cancel_futures=True)
    except TypeError:
        with contextlib.suppress(Exception):
            _EXECUTOR.shutdown(wait=True)
    except Exception:
        pass


_CANON_RE = re.compile(r"\s+")


def canon(s: str) -> str:
    """
    Canonicalize skill names: lowercase, strip, single-space.
    """
    return _CANON_RE.sub(" ", (s or "").strip().lower())


def build_skill_index(skill_tree) -> dict[str, str]:
    """
    Flatten level-3 skills into {canonical_name: original_name}
    so that we always match against what the user actually defined.
    Dynamic skill-tree friendly.
    """
    idx: dict[str, str] = {}

    if not getattr(skill_tree, "children", None):
        return idx

    for domain in skill_tree.children or []:
        for leaf in getattr(domain, "children", []) or []:
            # only leaf nodes (no further children)
            if getattr(leaf, "children", None):
                continue
            c = canon(leaf.name)
            if c:
                idx[c] = leaf.name
    return idx


def resolve_to_known_skill(raw: str, known: dict[str, str]) -> Optional[str]:
    """
    Try to map a focus-area string to one of the skill-tree leaves.

    Order:
      1. exact canonical match
      2. prefix match (either side) for small style diffs
      3. containment (subset) match

    This keeps the mapping explainable and avoids brittle LLM-only validation.
    """
    c = canon(raw)
    if not c:
        return None

    # 1) exact
    if c in known:
        return known[c]

    # 2) prefix
    for kc, orig in known.items():
        if c.startswith(kc) or kc.startswith(c):
            return orig

    # 3) containment
    for kc, orig in known.items():
        if kc in c or c in kc:
            return orig

    return None


def _classify_provider_error(exc: Exception) -> tuple[str, dict[str, Any]]:
    """
    Return (reason, extra) for structured error logging.
    """
    reason = "unknown"
    extra: dict[str, Any] = {"error": str(exc)}

    # asyncio timeout
    import asyncio as _asyncio

    if isinstance(exc, _asyncio.TimeoutError):
        return "timeout", extra

    # httpx
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

    # OpenAI/Anthropic SDK specific
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
    """
    Attach a per-thread TimedRotatingFileHandler writing to <log_dir>/<thread_id>/<agent_name>.log.
    """
    if not CFG.split_log_by_thread:
        return
    if not thread_id:
        return
    if thread_id in _THREAD_FILE_HANDLERS:
        return

    safe_tid = re.sub(r"[^\w.-]", "_", str(thread_id))
    thread_dir = os.path.join(CFG.log_dir, safe_tid)
    os.makedirs(thread_dir, exist_ok=True)

    log_filename = getattr(CFG, "log_file", f"{AGENT_NAME}.log")
    path = os.path.join(thread_dir, log_filename)

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
    """Close and remove the per-thread file handler if present."""
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
    """
    Configure and return the agent logger.
    """
    logger = logging.getLogger(AGENT_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(CFG.log_level)
    logger.propagate = False

    os.makedirs(CFG.log_dir, exist_ok=True)
    shared_path = os.path.join(CFG.log_dir, CFG.log_file)

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
            shared_path,
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
    """
    Set THREAD_ID_VAR from state.id, attach per-thread handler, restore afterwards.
    """

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


def _log_info(msg: str) -> None:
    LOGGER.info(msg)


def _log_warn(msg: str) -> None:
    LOGGER.warning(msg)


# ==============================
# Small JSON / logging helpers
# ==============================


def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return (t.startswith("{") and t.endswith("}")) or (
        t.startswith("[") and t.endswith("]")
    )


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
        return (
            json.dumps(value, ensure_ascii=False, indent=2)
            if isinstance(value, (dict, list))
            else str(value)
        )
    except Exception:
        return str(value)


def _pydantic_to_obj(obj: Any) -> Any:
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

            names: list[str] = []
            if isinstance(topics, list):
                for t in topics[:8]:
                    nm = None
                    if isinstance(t, dict):
                        nm = (
                            t.get("topic")
                            or t.get("name")
                            or t.get("title")
                            or t.get("label")
                        )
                    else:
                        try:
                            md = t.model_dump()
                            nm = (
                                md.get("topic")
                                or md.get("name")
                                or md.get("title")
                                or md.get("label")
                            )
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

    planned = getattr(ai_msg, "tool_calls", None)
    if planned:
        _log_info("Tool plan:")
        for tc in planned:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            LOGGER.info(f"  planned -> {name} args={_render_tool_payload(args)}")

    tool_msgs: list[Any] = []
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


def _log_feedback_event(action: str, text: str) -> None:
    if not CFG.log_feedbacks:
        return
    level = getattr(logging, CFG.feedback_log_level, logging.INFO)
    body = text or ""
    LOGGER.log(level, "Feedback %s (len=%d)", action, len(body))
    LOGGER.log(level, "%s", body)


def _log_feedback_cumulative(all_text: str) -> None:
    if not CFG.log_feedbacks:
        return
    level = getattr(logging, CFG.feedback_log_level, logging.INFO)
    body = all_text or ""
    LOGGER.log(level, "Cumulative feedbacks (len=%d)", len(body))
    LOGGER.log(level, "%s", body)


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
    try:
        reason, extra = _classify_provider_error(last_exc)
    except Exception:
        reason, extra = "unknown", {"error": str(last_exc)}

    LOGGER.error(
        "Terminal failure after %s attempt(s) | context=%s | reason=%s | extra=%s",
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
    Wrap a BaseTool with timeout + retries (both sync and async paths).
    """

    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff_base_s: float = PrivateAttr()

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
        self._backoff_base_s = backoff_base_s

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
                with contextlib.suppress(Exception):
                    future.cancel()
                last_exc = exc
                _log_retry(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                with contextlib.suppress(Exception):
                    future.cancel()
                last_exc = exc
                _log_retry(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
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
            return await loop.run_in_executor(
                _EXECUTOR,
                lambda: self._inner._run(*args, **{**kwargs, "config": config}),
            )

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

    _RAW_TOOLS: list[BaseTool] = get_mongo_tools(llm=llm)
    TOOLS: list[BaseTool] = [
        RetryTool(
            t,
            retries=CFG.tool_retries,
            timeout_s=CFG.tool_timeout_s,
            backoff_base_s=CFG.tool_backoff_base_s,
        )
        for t in _RAW_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(CollectiveInterviewTopicSchema)

    _compiled_inner_graph = None

    # ----- LLM invokers -----

    @staticmethod
    async def _invoke_agent(messages: Sequence[Msg]) -> Any:
        async def _call():
            if hasattr(TopicGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await TopicGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _EXECUTOR, TopicGenerationAgent._AGENT_MODEL.invoke, messages
            )

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
            return await loop.run_in_executor(
                _EXECUTOR, TopicGenerationAgent._STRUCTURED_MODEL.invoke, payload
            )

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
        _log_tool_activity(state["messages"], ai_msg=None)
        ai = await TopicGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _MongoAgentState):
        msgs = state["messages"]

        ai_content: Optional[str] = None
        for m in reversed(msgs):
            if getattr(m, "type", None) in ("ai", "assistant") and not getattr(
                m, "tool_calls", None
            ):
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
        g.add_conditional_edges(
            "agent", cls._should_continue, {"continue": "tools", "respond": "respond"}
        )
        g.add_edge("tools", "agent")
        g.add_edge("respond", END)

        cls._compiled_inner_graph = g.compile()
        return cls._compiled_inner_graph

    # ==========================
    # Outer graph: main node
    # ==========================

    @staticmethod
    @with_thread_context
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        fb = getattr(state, "interview_topics_feedback", None)
        if isinstance(fb, dict):
            feedback_text = fb.get("feedback", "")
        else:
            feedback_text = getattr(fb, "feedback", "") or ""

        if feedback_text:
            if state.interview_topics_feedbacks:
                state.interview_topics_feedbacks += "\n"
            state.interview_topics_feedbacks += feedback_text

            _log_feedback_event("appended", feedback_text)
            _log_feedback_cumulative(state.interview_topics_feedbacks)

        class AtTemplate(Template):
            delimiter = "@"

        sys_content = AtTemplate(TOPIC_GENERATION_AGENT_PROMPT).substitute(
            generated_summary=state.generated_summary.model_dump_json(),
            interview_topics_feedbacks=state.interview_topics_feedbacks,
            thread_id=state.id,
        )

        messages: list[Msg] = [
            SystemMessage(content=sys_content),
            HumanMessage(
                content="Based on the instructions, please start the process."
            ),
        ]

        inner_graph = TopicGenerationAgent._get_inner_graph()
        result = await inner_graph.ainvoke({"messages": messages})
        state.interview_topics = result["final_response"]

        rendered = _render_topics_for_log(
            state.interview_topics.model_dump_json(indent=2)
        )
        _log_info(f"Topics generated before validation | output={rendered}")
        return state

    # ==========================
    # Outer graph: router
    # ==========================

    @staticmethod
    @with_thread_context
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Validator without outer retries.

        1. Build index from the current skill tree.
        2. Normalize all topic focus_area skills to the skill-tree names.
        3. Fix total_questions in-place if needed.
        4. Inject missing MUST skills into the last topic's focus_area in-place.
        5. Return True to exit the graph.
        """
        # 1) dynamic skill index
        skill_index = build_skill_index(state.skill_tree)

        # 2) total questions check
        total_questions = sum(
            t.total_questions for t in state.interview_topics.interview_topics
        )
        target_questions = state.generated_summary.total_questions

        if total_questions != target_questions:
            LOGGER.warning(
                "Topic gen: total questions mismatch: got=%s want=%s. Adjusting last topic in-place.",
                total_questions,
                target_questions,
            )
            if state.interview_topics.interview_topics:
                last_topic = state.interview_topics.interview_topics[-1]
                last_topic.total_questions = max(
                    0,
                    target_questions - (total_questions - last_topic.total_questions),
                )

        # 3) normalize focus_area skills in-place
        normalized_focus: list[str] = []
        for topic in state.interview_topics.interview_topics:
            for fa in topic.focus_area:
                raw_skill = fa.model_dump().get("skill", "")
                resolved = resolve_to_known_skill(raw_skill, skill_index)
                if resolved:
                    fa.skill = resolved
                    normalized_focus.append(canon(resolved))
                else:
                    normalized_focus.append(canon(raw_skill))

        normalized_set = set(normalized_focus)

        # 4) collect MUST skills from tree
        must_skills: list[str] = []
        if getattr(state.skill_tree, "children", None):
            for domain in state.skill_tree.children or []:
                for leaf in getattr(domain, "children", []) or []:
                    if getattr(leaf, "children", None):
                        continue
                    if getattr(leaf, "priority", None) == "must":
                        must_skills.append(leaf.name)

        # 5) find missing MUSTs
        missing: list[str] = [
            ms for ms in must_skills if canon(ms) not in normalized_set
        ]

        if missing:
            logger = logging.getLogger(AGENT_NAME)
            logger.warning(
                "Topic gen: missing MUST skills, fixing in-place: %s", missing
            )

            if state.interview_topics.interview_topics:
                last_topic = state.interview_topics.interview_topics[-1]

                # try to infer FA class and its required fields
                fa_type = None
                if last_topic.focus_area:
                    fa_type = last_topic.focus_area[0].__class__
                else:
                    # look at earlier topics
                    for t in state.interview_topics.interview_topics:
                        if t.focus_area:
                            fa_type = t.focus_area[0].__class__
                            break

                if fa_type is None:
                    # we cannot safely construct a focus-area object, so just stop
                    return True

                # now actually append missing skills
                for ms in missing:
                    # check what fields the FA model expects
                    # pydantic v2: model_fields
                    fields = getattr(fa_type, "model_fields", {})  # {} if not pydantic
                    payload = {"skill": ms}

                    if "guideline" in fields:
                        payload["guideline"] = (
                            "Cover this skill in the interview. Injected from MUST list."
                        )

                    # add more defaults if your FA schema needs them:
                    # if "priority" in fields: payload["priority"] = "must"
                    # if "difficulty" in fields: payload["difficulty"] = "medium"

                    last_topic.focus_area.append(fa_type(**payload))

            # also write feedback back to state, like before
            feedback = FEEDBACK_HEADER + ", ".join(missing) + "\n"
            state.interview_topics_feedback = {
                "satisfied": False,
                "feedback": feedback,
            }
            if state.interview_topics_feedbacks:
                state.interview_topics_feedbacks += "\n" + feedback
            else:
                state.interview_topics_feedbacks = feedback

            _log_feedback_event("generated", feedback)
            _log_feedback_cumulative(state.interview_topics_feedbacks)

        LOGGER.info(
            "Topic generation completed after in-place normalization/fix | output=<hidden>"
        )
        return True

    # ==========================
    # Outer graph: builder
    # ==========================

    @staticmethod
    def get_graph(checkpointer=None):
        """
        START -> topic_generator -> should_regenerate -> END
        (no outer LLM retry; fixes are in-place)
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


# -------- Process-exit safety nets --------
@atexit.register
def _shutdown_topic_agent_at_exit() -> None:
    with contextlib.suppress(Exception):
        _close_all_thread_file_handlers()
    with contextlib.suppress(Exception):
        shutdown_executor()


if __name__ == "__main__":
    graph = TopicGenerationAgent.get_graph()
