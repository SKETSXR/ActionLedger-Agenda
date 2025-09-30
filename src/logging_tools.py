import json
import logging
import os
import sys
from datetime import datetime, date
from typing import Any, Optional, Sequence


def log_retry_iteration(
    agent_name: str,
    iteration: int,
    reason: str = "",
    *,
    logger: Optional[logging.Logger] = None,
    pretty_json: bool = False,
    extra: Optional[dict] = None,
) -> None:
    """
    Record a retry attempt for an agent, including the reason and any extra context.

    Args:
        agent_name: Logical name of the agent performing the retry.
        iteration: Monotonic retry counter for this agent flow.
        reason: Short reason for the retry decision.
        logger: Optional preconfigured logger; falls back to get_tool_logger(agent_name).
        pretty_json: If True, emit indented JSON for readability (larger logs).
        extra: Optional additional fields to include under the 'extra' key.
    """
    log = logger or get_tool_logger(agent_name)
    payload = {"iteration": iteration, "reason": reason}
    if extra:
        payload["extra"] = extra

    _emit(
        logger=log,
        event="retry_iteration",
        agent=agent_name,
        data=payload,
        pretty_json=pretty_json,
    )


def get_tool_logger(
    agent_name: str,
    log_dir: str = "logs",
    when: str = "midnight",  # kept for caller's compatibility; not used by FileHandler
    interval: int = 1,       # kept for caller's compatibility; not used by FileHandler
    backup_count: int = 365, # kept for caller's compatibility; not used by FileHandler
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a process-wide logger for a given agent, creating it on first use.

    The logger writes JSON lines to both stdout and a rotating file under `log_dir`.
    Rotation parameters are accepted for compatibility but not applied here because
    we use a basic FileHandler to preserve behavior.

    Args:
        agent_name: Logical agent name; also used to name the log file.
        log_dir: Directory where the log file is stored.
        when, interval, backup_count: Unused placeholders to preserve signature.
        level: Minimum log level.

    Returns:
        A configured Logger instance. Reuses existing handlers if already set up.
    """
    logger_name = f"tool_activity::{agent_name}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{agent_name}.log")

    # Console handler (stdout) for quick inspection in containerized runs
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(message)s"))

    # File handler (JSON lines)
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------- Safe getters for LangChain message-like objects ----------

def _msg_type(m: Any) -> Optional[str]:
    """Return message type if present (e.g., 'ai', 'human', 'tool'), else None."""
    return getattr(m, "type", None)


def _msg_content(m: Any) -> Any:
    """Return message content if present, else None."""
    return getattr(m, "content", None)


def _tool_call_id(m: Any) -> Any:
    """Return tool_call_id from a ToolMessage if present, else None."""
    return getattr(m, "tool_call_id", None)


# ---------- JSON normalizers (stable, safe, single-pass) ----------

def _maybe_parse_json(val: Any) -> Any:
    """
    If `val` is a JSON-looking string (object/array), attempt to parse once.
    Otherwise return it unchanged.

    This is useful when upstream tools serialize nested JSON into strings.
    """
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return val
    return val


def _json_sanitize(x: Any) -> Any:
    """
    Convert arbitrary Python objects into JSON-serializable structures:
      - Datetimes -> ISO8601 strings
      - Sets -> sorted lists (deterministic)
      - Pydantic models -> dicts via model_dump()/dict()
      - Generic objects -> vars(__dict__)
      - Fallback -> str(x)

    Also attempts to de-stringify nested JSON for common keys such as args/result.
    """
    # Primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # Sets -> sorted lists for deterministic output
    if isinstance(x, set):
        try:
            return sorted(x)
        except Exception:
            return list(x)

    # Lists / tuples
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]

    # Dicts
    if isinstance(x, dict):
        parsed = {}
        for k, v in x.items():
            if k in {"args", "result", "normalized_query", "query"}:
                v = _maybe_parse_json(v)
            parsed[str(k)] = _json_sanitize(v)
        return parsed

    # Datetimes
    if isinstance(x, (datetime, date)):
        return x.isoformat()

    # Pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return _json_sanitize(x.model_dump())
        except Exception:
            pass

    # Pydantic v1
    if hasattr(x, "dict"):
        try:
            return _json_sanitize(x.dict())
        except Exception:
            pass

    # Generic objects
    if hasattr(x, "__dict__"):
        return _json_sanitize(vars(x))

    # Final fallback
    return str(x)


# ---------- Emit helper (unified interface) ----------

def _emit(
    logger: logging.Logger,
    *,
    event: str,
    agent: str,
    data: Any,
    ts: Optional[str] = None,
    pretty_json: bool = False,
) -> None:
    """
    Emit a single JSON log record with a consistent envelope.

    Args:
        logger: Destination logger instance.
        event: Short event type (e.g., 'tool_call_planned', 'tool_result').
        agent: Agent name for scoping.
        data: Arbitrary payload; will be sanitized for JSON.
        ts: Optional timestamp override; defaults to now in ISO8601.
        pretty_json: If True, indent JSON for readability.

    Behavior:
        - Performs a single sanitize pass so nested strings holding JSON are parsed.
        - Emits one line per record (pretty or compact).
    """
    record = {
        "event": event,
        "agent": agent,
        "ts": ts or datetime.now().isoformat(timespec="seconds"),
        "data": _json_sanitize(data),
    }
    if pretty_json:
        logger.info(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


# ---------- Public interface: unified activity logger ----------

def log_tool_activity(
    messages: Sequence[Any],
    ai_msg: Optional[Any] = None,
    agent_name: str = "agent",
    logger: Optional[logging.Logger] = None,
    header: str = "Tool Activity",
    pretty_json: bool = False,
) -> None:
    """
    Log planned tool calls (from the latest assistant message) and any recent
    tool results located at the tail of `messages`.

    Args:
        messages: Conversation transcript; recent tool results are expected at the tail.
        ai_msg: The assistant message that potentially contains planned tool calls.
        agent_name: Agent identifier for the log stream and file naming.
        logger: Optional preconfigured logger; falls back to get_tool_logger(agent_name).
        header: Human-readable section title used in banner events.
        pretty_json: If True, emit indented JSON.

    Notes:
        - Only examines trailing ToolMessages for results to avoid scanning entire history.
        - Uses a consistent 'banner' / 'banner_end' pair to demarcate sections.
    """
    log = logger or get_tool_logger(agent_name)
    now = datetime.now().isoformat(timespec="seconds")

    # Planned tool calls (from the assistant's message)
    tool_calls = getattr(ai_msg, "tool_calls", None)
    if tool_calls:
        _emit(
            logger=log,
            event="banner",
            agent=agent_name,
            data=f"{header}: planned",
            ts=now,
            pretty_json=pretty_json,
        )

        for tc in tool_calls or []:
            try:
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args")
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", None)
            except Exception:
                name, args = str(tc), None

            _emit(
                logger=log,
                event="tool_call_planned",
                agent=agent_name,
                data={"name": name, "args": args},
                ts=now,
                pretty_json=pretty_json,
            )

        _emit(
            logger=log,
            event="banner_end",
            agent=agent_name,
            data="end",
            ts=now,
            pretty_json=pretty_json,
        )

    # Recent tool results (walk backward while tail messages are tool outputs)
    i = len(messages) - 1
    printed_results = False
    while i >= 0 and _msg_type(messages[i]) == "tool":
        if not printed_results:
            _emit(
                logger=log,
                event="banner",
                agent=agent_name,
                data=f"{header}: results",
                ts=now,
                pretty_json=pretty_json,
            )
            printed_results = True

        tm = messages[i]
        _emit(
            logger=log,
            event="tool_result",
            agent=agent_name,
            data={"tool_call_id": _tool_call_id(tm), "result": _msg_content(tm)},
            ts=now,
            pretty_json=pretty_json,
        )
        i -= 1

    if printed_results:
        _emit(
            logger=log,
            event="banner_end",
            agent=agent_name,
            data="end",
            ts=now,
            pretty_json=pretty_json,
        )
