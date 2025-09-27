# """
# Reusable tool-call logger for LangGraph/LangChain agents.

# Features:
# - Unified logger for planned tool calls and recent tool results
# - Logs to console + per-agent rotating file (daily rotation, keep 365 days)
# - JSONL-style entries for easy parsing
# - Safe against dict/attr shape differences in LangChain versions
# """

# import json
# import logging
# import os
# import sys
# from datetime import datetime
# from logging.handlers import TimedRotatingFileHandler
# from typing import Any, Optional, Sequence

# # ---------- Logger setup ----------

# def get_tool_logger(
#     agent_name: str,
#     log_dir: str = "logs",
#     when: str = "midnight",
#     interval: int = 1,
#     backup_count: int = 365,
#     level: int = logging.INFO,
# ) -> logging.Logger:
#     """
#     Create or fetch a logger that writes to console and to a per-agent rotating file.

#     Rotation policy:
#       - Time-based rotation ('when' and 'interval') with TimedRotatingFileHandler
#       - backup_count=365 keeps about a year of daily logs (oldest files removed automatically)
#     """
#     logger_name = f"tool_activity::{agent_name}"
#     logger = logging.getLogger(logger_name)
#     if logger.handlers:
#         return logger  # already configured

#     logger.setLevel(level)
#     logger.propagate = False  # avoid duplicate prints

#     os.makedirs(log_dir, exist_ok=True)
#     file_path = os.path.join(log_dir, f"{agent_name}.log")

#     # Console handler
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(level)
#     ch.setFormatter(logging.Formatter("%(message)s"))

#     # File handler with time-based rotation and retention
#     fh = TimedRotatingFileHandler(
#         file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8"
#     )
#     # If you prefer UTC-based rotation, uncomment:
#     # fh.utc = True
#     fh.setLevel(level)
#     fh.setFormatter(logging.Formatter("%(message)s"))

#     logger.addHandler(ch)
#     logger.addHandler(fh)
#     return logger


# # ---------- Safe getters for LC message objects ----------

# def _msg_type(m: Any) -> Optional[str]:
#     return getattr(m, "type", None)

# def _msg_content(m: Any) -> Any:
#     return getattr(m, "content", None)

# def _tool_call_id(m: Any) -> Any:
#     return getattr(m, "tool_call_id", None)


# # ---------- Public API: unified activity logger ----------

# def log_tool_activity(
#     messages: Sequence[Any],
#     ai_msg: Optional[Any] = None,
#     agent_name: str = "agent",
#     logger: Optional[logging.Logger] = None,
#     header: str = "Tool Activity",
# ) -> None:
#     """
#     Logs planned tool calls from the assistant message (ai_msg) and
#     recent tool results found at the tail of `messages`.

#     Writes JSONL events with keys:
#       event, agent, ts, data
#     """
#     log = logger or get_tool_logger(agent_name)
#     now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

#     # 1) Planned tool calls
#     tool_calls = getattr(ai_msg, "tool_calls", None)
#     if tool_calls:
#         log.info(json.dumps({
#             "event": "banner",
#             "agent": agent_name,
#             "ts": now,
#             "data": f"{header}: planned"
#         }))

#         for tc in tool_calls or []:
#             # tolerate dict or object
#             name = None
#             args = None
#             try:
#                 if isinstance(tc, dict):
#                     name = tc.get("name")
#                     args = tc.get("args")
#                 else:
#                     name = getattr(tc, "name", None)
#                     args = getattr(tc, "args", None)
#             except Exception:
#                 name = str(tc)
#                 args = None

#             log.info(json.dumps({
#                 "event": "tool_call_planned",
#                 "agent": agent_name,
#                 "ts": now,
#                 "data": {"name": name, "args": args}
#             }))

#     # 2) Recent tool results (walk back while trailing messages are of type 'tool')
#     i = len(messages) - 1
#     has_results = False
#     while i >= 0 and _msg_type(messages[i]) == "tool":
#         if not has_results:
#             log.info(json.dumps({
#                 "event": "banner",
#                 "agent": agent_name,
#                 "ts": now,
#                 "data": f"{header}: results"
#             }))
#             has_results = True

#         tm = messages[i]
#         log.info(json.dumps({
#             "event": "tool_result",
#             "agent": agent_name,
#             "ts": now,
#             "data": {
#                 "tool_call_id": _tool_call_id(tm),
#                 "result": _msg_content(tm)
#             }
#         }))
#         i -= 1

#     if tool_calls or has_results:
#         log.info(json.dumps({
#             "event": "banner_end",
#             "agent": agent_name,
#             "ts": now,
#             "data": "end"
#         }))


import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Any, Optional, Sequence

# ---------- Logger setup ----------


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
    Log a retry iteration (to console + file).
    - agent_name: e.g., "discussion_summary_agent"
    - iteration: your module-level `count` (before or after increment, your call)
    - reason: short human-readable reason or phase label
    - extra: optional structured context (e.g., {"missing": [...], "schema_errors": "..."})
    """
    log = logger or get_tool_logger(agent_name)
    payload = {"iteration": iteration, "reason": reason}
    if extra:
        payload["extra"] = extra

    _emit(
        logger=log,
        event="retry_iteration",
        agent_name=agent_name,
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
    Create or fetch a logger that writes to console and to a per-agent rotating file.

    Rotation policy:
      - Time-based rotation ('when' and 'interval') with TimedRotatingFileHandler
      - backup_count=365 keeps ~1 year of daily logs (oldest files auto-removed)
    """
    logger_name = f"tool_activity::{agent_name}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate prints

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, f"{agent_name}.log")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(message)s"))

    # File handler with rotation + retention
    fh = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# ---------- Safe getters for LC message objects ----------

def _msg_type(m: Any) -> Optional[str]:
    return getattr(m, "type", None)


def _msg_content(m: Any) -> Any:
    return getattr(m, "content", None)


def _tool_call_id(m: Any) -> Any:
    return getattr(m, "tool_call_id", None)

# ---------- Emit helper ----------

def _emit(
    logger: logging.Logger,
    event: str,
    agent_name: str,
    data: Any,
    ts: Optional[str] = None,
    pretty_json: bool = False,
) -> None:
    obj = {
        "event": event,
        "agent": agent_name,
        "ts": ts or (datetime.utcnow().isoformat(timespec="seconds") + "Z"),
        "data": data,
    }
    if pretty_json:
        logger.info(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        logger.info(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

# ---------- Public API: unified activity logger ----------

def log_tool_activity(
    messages: Sequence[Any],
    ai_msg: Optional[Any] = None,
    agent_name: str = "agent",
    logger: Optional[logging.Logger] = None,
    header: str = "Tool Activity",
    pretty_json: bool = False,  # <<< enable pretty printing
) -> None:
    """
    Logs planned tool calls from the assistant message (ai_msg) and
    recent tool results found at the tail of `messages`.

    Set `pretty_json=True` for indented, multi-line JSON output.
    """
    log = logger or get_tool_logger(agent_name)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Planned tool calls
    tool_calls = getattr(ai_msg, "tool_calls", None)
    if tool_calls:
        _emit(log, "banner", agent_name, f"{header}: planned", ts=now, pretty_json=pretty_json)

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
                log,
                "tool_call_planned",
                agent_name,
                {"name": name, "args": args},
                ts=now,
                pretty_json=pretty_json,
            )

        _emit(log, "banner_end", agent_name, "end", ts=now, pretty_json=pretty_json)

    # Recent tool results
    i = len(messages) - 1
    printed_results = False
    while i >= 0 and _msg_type(messages[i]) == "tool":
        if not printed_results:
            _emit(log, "banner", agent_name, f"{header}: results", ts=now, pretty_json=pretty_json)
            printed_results = True

        tm = messages[i]
        _emit(
            log,
            "tool_result",
            agent_name,
            {"tool_call_id": _tool_call_id(tm), "result": _msg_content(tm)},
            ts=now,
            pretty_json=pretty_json,
        )
        i -= 1

    if printed_results:
        _emit(log, "banner_end", agent_name, "end", ts=now, pretty_json=pretty_json)
