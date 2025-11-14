# =============================================================================
# Module: summary_generation_agent
# =============================================================================
# Purpose
#   Orchestrate one LLM call to produce a structured interview summary from:
#     • job_description
#     • skill_tree
#     • candidate_profile
#
# Responsibilities
#   • Build deterministic System + Human messages from AgentInternalState.
#   • Invoke LLM with structured output (GeneratedSummarySchema).
#   • Enforce timeout and bounded exponential-backoff retries.
#   • Store the result in state.generated_summary.
#   • Emit JSON-style per thread id logs to file and concise logs to console.
#
# Data Flow
#   START ──► summary_generator (async) ──► END
#     1) Validate required state fields.
#     2) Render prompt with JSON-serialized inputs.
#     3) Call LLM (structured output, timeout, retries).
#     4) Write GeneratedSummarySchema to state.
#
# Interface
#   • SummaryGenerationAgent.summary_generator(state) -> state
#   • SummaryGenerationAgent.get_graph(checkpointer=None) -> Compiled graph
#
# Config (via environment variables)
#   SUMMARY_AGENT_LOG_DIR                     (default: logs)
#   SUMMARY_AGENT_LOG_FILE                    (default: summary_generation_agent.log)
#   SUMMARY_AGENT_LOG_LEVEL_CONSOLE           (default: INFO)
#   SUMMARY_AGENT_LOG_LEVEL_FILE              (default: INFO)
#   SUMMARY_AGENT_LOG_BACKUP_DAYS             (default: 365)
#   SUMMARY_AGENT_LLM_TIMEOUT_SECONDS         (default: 90)
#   SUMMARY_AGENT_LLM_RETRIES                 (default: 2)
#   SUMMARY_AGENT_LLM_RETRY_BACKOFF_SECONDS   (default: 2.5)
#   SUMMARY_AGENT_RESULT_LOG_PAYLOAD          off | summary | full   (default: off)
#   SUMMARY_AGENT_LOG_SPLIT_BY_THREAD         (0|1)
#
# Notes
#   • Inputs are JSON-serialized to reduce prompt drift.
#   • LLM client is injected (../model_handling.llm_sg) for testability.
# =============================================================================

import asyncio
import atexit
import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import re
import sys
from typing import Optional, Sequence, Union
import uuid

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph

from src.model_handling import llm_sg as _llm_client
from src.prompt.summary_generation_agent_prompt import SUMMARY_GENERATION_AGENT_PROMPT
from src.schema.agent_schema import AgentInternalState
from src.schema.output_schema import GeneratedSummarySchema


# =============================================================================
# Configuration (env-driven, safe defaults)
# =============================================================================


@dataclass(frozen=True)
class SummaryAgentConfig:
    # Logging
    log_dir: str = os.getenv("SUMMARY_AGENT_LOG_DIR", "logs")
    log_file: str = os.getenv("SUMMARY_AGENT_LOG_FILE", "summary_generation_agent.log")
    log_level_console: str = os.getenv("SUMMARY_AGENT_LOG_LEVEL_CONSOLE", "INFO")
    log_level_file: str = os.getenv("SUMMARY_AGENT_LOG_LEVEL_FILE", "INFO")
    log_backup_days: int = int(os.getenv("SUMMARY_AGENT_LOG_BACKUP_DAYS", "365"))

    # LLM call behavior
    llm_timeout_seconds: float = float(
        os.getenv("SUMMARY_AGENT_LLM_TIMEOUT_SECONDS", "90")
    )
    llm_retries: int = int(os.getenv("SUMMARY_AGENT_LLM_RETRIES", "2"))
    llm_retry_backoff_seconds: float = float(
        os.getenv("SUMMARY_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5")
    )

    # Output logging: off | summary | full
    result_log_payload: str = (
        os.getenv("SUMMARY_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()
    )

    # Error verbosity: include traceback if true
    include_stacks: bool = os.getenv(
        "SUMMARY_AGENT_INCLUDE_STACKS", "false"
    ).strip().lower() in {"1", "true", "yes"}
    # Split logs by thread id (default on). Set to 0/false for a single shared file.
    split_log_by_thread: bool = os.getenv(
        "SUMMARY_AGENT_LOG_SPLIT_BY_THREAD", "1"
    ).strip().lower() in ("1", "true", "yes", "y")


# Current thread id for logging context (used by all log records)
THREAD_ID_VAR: ContextVar[str] = ContextVar("thread_id", default="-")


class _ThreadIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.thread_id = THREAD_ID_VAR.get()
        except Exception:
            record.thread_id = "-"
        return True


# Avoid duplicate handlers per thread id
_THREAD_FILE_HANDLERS: dict[str, logging.Handler] = {}


def _attach_thread_file_handler(thread_id: str) -> None:
    """Attach a per-thread TimedRotatingFileHandler writing to <log_dir>/<thread_id>/<log_file>."""
    if not CONFIG.split_log_by_thread or not thread_id:
        return
    if thread_id in _THREAD_FILE_HANDLERS:
        return

    # <log_dir>/<thread_id>/summary_generation_agent.log  (folder per thread)
    safe_tid = re.sub(r"[^\w.-]", "_", str(thread_id))  # requires: import re
    thread_dir = os.path.join(CONFIG.log_dir, safe_tid)
    os.makedirs(thread_dir, exist_ok=True)
    path = os.path.join(thread_dir, CONFIG.log_file)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | tid=%(thread_id)s | %(message)s"
    )
    handler = TimedRotatingFileHandler(
        filename=path,
        when="midnight",
        interval=1,
        backupCount=CONFIG.log_backup_days,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    logging.raiseExceptions = False
    handler.setLevel(getattr(logging, CONFIG.log_level_file.upper(), logging.INFO))
    handler.setFormatter(fmt)
    handler.set_name(f"file::thread::{safe_tid}")
    handler.addFilter(_ThreadIdFilter())

    LOGGER.addHandler(handler)
    _THREAD_FILE_HANDLERS[thread_id] = handler


def _detach_thread_file_handler(thread_id: str) -> None:
    """Close and remove a per-thread handler if present (best-effort)."""
    h = _THREAD_FILE_HANDLERS.pop(thread_id, None)
    if h:
        try:
            LOGGER.removeHandler(h)
        finally:
            with contextlib.suppress(Exception):
                h.close()


def _close_all_thread_file_handlers() -> None:
    """Close all per-thread handlers on process exit (best-effort)."""
    for tid in list(_THREAD_FILE_HANDLERS.keys()):
        _detach_thread_file_handler(tid)


CONFIG = SummaryAgentConfig()


# =============================================================================
# Logger (rotating file + console; never crash on logging errors)
# =============================================================================


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass  # best effort


def _get_logger(name: str = "summary_generation_agent") -> logging.Logger:
    """Return a configured, idempotent logger for this agent.

    Adds a stdout handler (level from CONFIG.log_level_console) and, when
    `CONFIG.split_log_by_thread` is False, a timed-rotating file handler
    (level from CONFIG.log_level_file). Both use a format that injects a
    `thread_id` via `_ThreadIdFilter`. Disables propagation/raiseExceptions
    and guards re-init with a private `_initialized` flag.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(logging.DEBUG)
    _ensure_dir(CONFIG.log_dir)

    # Formatter that includes tid
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | tid=%(thread_id)s | %(message)s"
    )
    tid_filter = _ThreadIdFilter()

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(
        getattr(logging, CONFIG.log_level_console.upper(), logging.INFO)
    )
    console_handler.setFormatter(fmt)
    console_handler.addFilter(tid_filter)
    logger.addHandler(console_handler)

    # Shared file handler only if per-thread split is OFF
    if not CONFIG.split_log_by_thread:
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(CONFIG.log_dir, CONFIG.log_file),
            when="midnight",
            interval=1,
            backupCount=CONFIG.log_backup_days,
            encoding="utf-8",
            utc=False,
            delay=True,
        )
        file_handler.setLevel(
            getattr(logging, CONFIG.log_level_file.upper(), logging.INFO)
        )
        file_handler.setFormatter(fmt)
        file_handler.set_name("file::shared")
        file_handler.addFilter(tid_filter)
        logger.addHandler(file_handler)

    logging.raiseExceptions = False
    logger.propagate = False
    logger._initialized = True  # type: ignore[attr-defined]
    return logger


def with_thread_context(fn):
    """Set THREAD_ID_VAR from state.id, attach per-thread handler, then restore."""

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


# =============================================================================
# Minimal, safe serialization helpers (for logs only)
# =============================================================================


def _safe_dump_for_log(obj: object) -> str:
    """Best-effort JSON for logs. Handles Pydantic v1/v2 and plain objects."""
    # Try pydantic JSON first
    for attr in ("model_dump_json", "json"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    # Try pydantic dict, then JSON
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try:
                return json.dumps(getattr(obj, attr)(), ensure_ascii=False)
            except Exception:
                pass
    # Last resort
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "<unserializable>"


def _render_for_log(payload: object) -> str:
    """off | summary | full"""
    mode = CONFIG.result_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        try:
            # Try to peek top-level keys without materializing full payload
            if hasattr(payload, "model_dump"):
                data = payload.model_dump()
            elif hasattr(payload, "dict"):
                data = payload.dict()
            else:
                data = payload if isinstance(payload, dict) else None

            if isinstance(data, dict):
                keys = list(data.keys())
                return f"keys={keys[:8]}{'...' if len(keys) > 8 else ''}"
            if isinstance(payload, list):
                return f"list(len={len(payload)})"
            return type(payload).__name__
        except Exception:
            return "<unavailable>"
    # full
    return _safe_dump_for_log(payload)


# =============================================================================
# Prompt + LLM call
# =============================================================================

Msg = Union[SystemMessage, HumanMessage]


def build_messages(state: AgentInternalState) -> list[Msg]:
    """Deterministic prompt construction; JSON inputs to keep formatting stable."""
    system = SystemMessage(
        content=SUMMARY_GENERATION_AGENT_PROMPT.format(
            job_description=state.job_description.model_dump_json(),
            skill_tree=state.skill_tree.model_dump_json(),
            candidate_profile=state.candidate_profile.model_dump_json(),
        )
    )
    human = HumanMessage(
        content="Begin the summary generation per the instructions and input payload."
    )
    return [system, human]


def _brief_exception(exc: Exception) -> dict:
    """Compact, structured error without traceback."""
    out = {"type": exc.__class__.__name__, "message": (str(exc) or "").strip()}
    # Enrich common cases
    try:
        if isinstance(exc, httpx.HTTPStatusError):
            out["status_code"] = exc.response.status_code
            try:
                body = exc.response.json() or {}
                code = (body.get("error") or {}).get("code")
                if code:
                    out["code"] = code
            except Exception:
                pass
        elif isinstance(exc, asyncio.TimeoutError):
            out["code"] = "timeout"
    except Exception:
        pass
    if len(out.get("message", "")) > 300:
        out["message"] = out["message"][:300] + "…"
    return out


def _classify_error(exc: Exception) -> tuple[str, str]:
    """Return (event, reason) where event ∈ {llm_error, schema_error} for metrics."""
    if isinstance(exc, asyncio.TimeoutError):
        return "llm_error", "timeout"
    try:
        if isinstance(exc, httpx.HTTPStatusError):
            s = exc.response.status_code
            if s == 401:
                return "llm_error", "auth"
            if s == 403:
                return "llm_error", "permission"
            if s == 429:
                try:
                    body = exc.response.json() or {}
                    code = (body.get("error") or {}).get("code")
                    return (
                        ("llm_error", "quota")
                        if code == "insufficient_quota"
                        else ("llm_error", "rate_limited")
                    )
                except Exception:
                    return "llm_error", "rate_limited"
            if 400 <= s < 500:
                return "llm_error", "bad_request"
            if s >= 500:
                return "llm_error", "server"
    except Exception:
        pass
    # LangChain/Pydantic parsing/validation → schema_error
    try:
        from langchain.schema import OutputParserException

        if isinstance(exc, OutputParserException):
            return "schema_error", "parse"
    except Exception:
        pass
    for mod, name in (
        ("pydantic_core", "ValidationError"),
        ("pydantic", "ValidationError"),
    ):
        try:
            VE = getattr(__import__(mod, fromlist=[name]), name)
            if isinstance(exc, VE):
                return "schema_error", "pydantic"
        except Exception:
            pass
    return "llm_error", "unknown"


async def invoke_llm_with_retry(
    messages: Sequence[Msg], request_id: str
) -> GeneratedSummarySchema:
    """Call the LLM with timeout + exponential backoff retries. Return structured output."""
    last_error: Optional[Exception] = None

    for attempt in range(CONFIG.llm_retries + 1):
        try:
            LOGGER.info(
                "LLM call start",
                extra={
                    "request_id": request_id,
                    "event": "llm_call_start",
                    "attempt": attempt + 1,
                },
            )

            coro = _llm_client.with_structured_output(GeneratedSummarySchema).ainvoke(
                messages
            )
            result: GeneratedSummarySchema = await asyncio.wait_for(
                coro, timeout=CONFIG.llm_timeout_seconds
            )

            LOGGER.info(
                "LLM call success",
                extra={
                    "request_id": request_id,
                    "event": "llm_call_success",
                    "attempt": attempt + 1,
                },
            )
            return result

        except Exception as exc:
            last_error = exc
            event, reason = _classify_error(exc)
            LOGGER.warning(
                "LLM call failed",
                extra={
                    "request_id": request_id,
                    "event": event,
                    "reason": reason,
                    "attempt": attempt + 1,
                    "error": _brief_exception(exc),
                },
                exc_info=CONFIG.include_stacks,
            )

        if attempt < CONFIG.llm_retries:
            await asyncio.sleep(CONFIG.llm_retry_backoff_seconds * (2**attempt))

    # Terminal failure
    assert last_error is not None
    event, reason = _classify_error(last_error)
    LOGGER.error(
        "LLM call failed after retries",
        extra={
            "request_id": request_id,
            "event": event,
            "reason": reason,
            "attempts": CONFIG.llm_retries + 1,
            "error": _brief_exception(last_error),
        },
        exc_info=CONFIG.include_stacks,
    )
    raise last_error


# =============================================================================
# Agent
# =============================================================================


class SummaryGenerationAgent:
    """
    Build messages from state and produce GeneratedSummarySchema at state.generated_summary.

    Required fields on state:
      - job_description
      - skill_tree
      - candidate_profile
    """

    llm_client = _llm_client  # easy swapping/mocking

    @staticmethod
    @with_thread_context
    async def summary_generator(state: AgentInternalState) -> AgentInternalState:
        request_id = getattr(state, "request_id", None) or str(uuid.uuid4())
        node = "summary_generator"

        # Input guardrails
        for field in ("job_description", "skill_tree", "candidate_profile"):
            if getattr(state, field, None) is None:
                LOGGER.error(
                    "Missing required field on state: %s",
                    field,
                    extra={
                        "request_id": request_id,
                        "node": node,
                        "event": "state_validation_error",
                    },
                )
                return state  # keep flow alive; flip to raise if you want fail-fast

        messages = build_messages(state)

        try:
            summary = await invoke_llm_with_retry(messages, request_id)
            state.generated_summary = summary

            LOGGER.info(
                f"Summary generation completed | output={_render_for_log(summary)}",
                extra={
                    "request_id": request_id,
                    "node": node,
                    "event": "summary_generated",
                },
            )
        except Exception as exc:
            LOGGER.error(
                "Summary generation failed",
                extra={
                    "request_id": request_id,
                    "node": node,
                    "event": "summary_failed",
                    "error": _brief_exception(exc),
                },
                exc_info=CONFIG.include_stacks,
            )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        g = StateGraph(state_schema=AgentInternalState)
        g.add_node("summary_generator", SummaryGenerationAgent.summary_generator)
        g.add_edge(START, "summary_generator")
        return g.compile(checkpointer=checkpointer, name="Summary Generation Agent")


# =============================================================================
# CLI
# =============================================================================


# -------- Process-exit safety net (no behavior change to main flow) --------
@atexit.register
def _shutdown_summary_agent_at_exit() -> None:
    with contextlib.suppress(Exception):
        _close_all_thread_file_handlers()


if __name__ == "__main__":
    try:
        graph = SummaryGenerationAgent.get_graph()
    except Exception as exc:
        LOGGER.error(
            "Unable to draw ASCII graph.",
            extra={"error": _brief_exception(exc)},
            exc_info=CONFIG.include_stacks,
        )
