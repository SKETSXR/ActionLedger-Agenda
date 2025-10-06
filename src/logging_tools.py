import json
import logging
import os
import sys
from datetime import datetime, date
from logging.handlers import TimedRotatingFileHandler
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
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = 365,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a process-wide logger for a given agent, creating it on first use.

    Writes JSON lines to stdout and to a time-rotating file under `log_dir`.
    Subsequent calls reuse the same configured logger.
    """
    logger_name = f"tool_activity::{agent_name}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{agent_name}.log")

    # Console handler (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(message)s"))

    # Timed rotating file handler
    fh = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8", utc=False
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------- Safe getters for LangChain message-like objects ----------

def _msg_type(m: Any) -> Optional[str]:
    return getattr(m, "type", None)


def _msg_content(m: Any) -> Any:
    return getattr(m, "content", None)


def _tool_call_id(m: Any) -> Any:
    return getattr(m, "tool_call_id", None)


# ---------- JSON normalizers ----------

def _maybe_parse_json(val: Any) -> Any:
    """
    If `val` is a JSON-looking string (object/array), attempt to parse once.
    Otherwise return it unchanged.
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
      - Sets -> sorted lists
      - Pydantic models -> dicts via model_dump()/dict()
      - Generic objects -> vars(__dict__)
      - Fallback -> str(x)

    Also de-stringifies nested JSON for common keys like args/result.
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, set):
        try:
            return sorted(x)
        except Exception:
            return list(x)

    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]

    if isinstance(x, dict):
        parsed = {}
        for k, v in x.items():
            if k in {"args", "result", "normalized_query", "query"}:
                v = _maybe_parse_json(v)
            parsed[str(k)] = _json_sanitize(v)
        return parsed

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

    if hasattr(x, "__dict__"):
        return _json_sanitize(vars(x))

    return str(x)


# ---------- Emit helper ----------

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
    Log planned tool calls (from the latest assistant message) and recent tool
    results found at the end of `messages`.
    """
    if messages is None:
        return

    log = logger or get_tool_logger(agent_name)
    now = datetime.now().isoformat(timespec="seconds")

    # Planned tool calls
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

    # Recent tool results (walk backward while trailing messages are tool outputs)
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
