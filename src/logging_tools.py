# # Running but with \n
# import json
# import logging
# import os
# import sys
# from datetime import datetime
# from logging.handlers import TimedRotatingFileHandler
# from typing import Any, Optional, Sequence

# # ---------- Logger setup ----------


# def log_retry_iteration(
#     agent_name: str,
#     iteration: int,
#     reason: str = "",
#     *,
#     logger: Optional[logging.Logger] = None,
#     pretty_json: bool = False,
#     extra: Optional[dict] = None,
# ) -> None:
#     """
#     Log a retry iteration (to console + file).
#     - agent_name: e.g., "discussion_summary_agent"
#     - iteration: your module-level `count` (before or after increment, your call)
#     - reason: short human-readable reason or phase label
#     - extra: optional structured context (e.g., {"missing": [...], "schema_errors": "..."})
#     """
#     log = logger or get_tool_logger(agent_name)
#     payload = {"iteration": iteration, "reason": reason}
#     if extra:
#         payload["extra"] = extra

#     _emit(
#         logger=log,
#         event="retry_iteration",
#         agent_name=agent_name,
#         data=payload,
#         pretty_json=pretty_json,
#     )


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
#       - backup_count=365 keeps ~1 year of daily logs (oldest files auto-removed)
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

#     # File handler with rotation + retention
#     fh = TimedRotatingFileHandler(
#         file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8"
#     )
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

# # ---------- Emit helper ----------

# def _emit(
#     logger: logging.Logger,
#     event: str,
#     agent_name: str,
#     data: Any,
#     ts: Optional[str] = None,
#     pretty_json: bool = False,
# ) -> None:
#     obj = {
#         "event": event,
#         "agent": agent_name,
#         "ts": ts or (datetime.now().isoformat(timespec="seconds")),
#         "data": data,
#     }
#     if pretty_json:
#         logger.info(json.dumps(obj, ensure_ascii=False, indent=2))
#     else:
#         logger.info(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

# # ---------- Public API: unified activity logger ----------

# def log_tool_activity(
#     messages: Sequence[Any],
#     ai_msg: Optional[Any] = None,
#     agent_name: str = "agent",
#     logger: Optional[logging.Logger] = None,
#     header: str = "Tool Activity",
#     pretty_json: bool = False,  # <<< enable pretty printing
# ) -> None:
#     """
#     Logs planned tool calls from the assistant message (ai_msg) and
#     recent tool results found at the tail of `messages`.

#     Set `pretty_json=True` for indented, multi-line JSON output.
#     """
#     log = logger or get_tool_logger(agent_name)
#     now = datetime.now().isoformat(timespec="seconds")

#     # Planned tool calls
#     tool_calls = getattr(ai_msg, "tool_calls", None)
#     if tool_calls:
#         _emit(log, "banner", agent_name, f"{header}: planned", ts=now, pretty_json=pretty_json)

#         for tc in tool_calls or []:
#             try:
#                 if isinstance(tc, dict):
#                     name = tc.get("name")
#                     args = tc.get("args")
#                 else:
#                     name = getattr(tc, "name", None)
#                     args = getattr(tc, "args", None)
#             except Exception:
#                 name, args = str(tc), None

#             _emit(
#                 log,
#                 "tool_call_planned",
#                 agent_name,
#                 {"name": name, "args": args},
#                 ts=now,
#                 pretty_json=pretty_json,
#             )

#         _emit(log, "banner_end", agent_name, "end", ts=now, pretty_json=pretty_json)

#     # Recent tool results
#     i = len(messages) - 1
#     printed_results = False
#     while i >= 0 and _msg_type(messages[i]) == "tool":
#         if not printed_results:
#             _emit(log, "banner", agent_name, f"{header}: results", ts=now, pretty_json=pretty_json)
#             printed_results = True

#         tm = messages[i]
#         _emit(
#             log,
#             "tool_result",
#             agent_name,
#             {"tool_call_id": _tool_call_id(tm), "result": _msg_content(tm)},
#             ts=now,
#             pretty_json=pretty_json,
#         )
#         i -= 1

#     if printed_results:
#         _emit(log, "banner_end", agent_name, "end", ts=now, pretty_json=pretty_json)

# # test

# import json
# import logging
# import os
# import sys
# from datetime import datetime
# from logging.handlers import TimedRotatingFileHandler
# from typing import Any, Optional, Sequence

# # ---------- Logger setup ----------

# def log_retry_iteration(
#     agent_name: str,
#     iteration: int,
#     reason: str = "",
#     *,
#     logger: Optional[logging.Logger] = None,
#     pretty_json: bool = False,
#     extra: Optional[dict] = None,
# ) -> None:
#     log = logger or get_tool_logger(agent_name)
#     payload = {"iteration": iteration, "reason": reason}
#     if extra:
#         payload["extra"] = extra

#     _emit(
#         logger=log,
#         event="retry_iteration",
#         agent_name=agent_name,
#         data=payload,
#         pretty_json=pretty_json,
#     )


# def get_tool_logger(
#     agent_name: str,
#     log_dir: str = "logs",
#     when: str = "midnight",
#     interval: int = 1,
#     backup_count: int = 365,
#     level: int = logging.INFO,
# ) -> logging.Logger:
#     logger_name = f"tool_activity::{agent_name}"
#     logger = logging.getLogger(logger_name)
#     if logger.handlers:
#         return logger

#     logger.setLevel(level)
#     logger.propagate = False

#     os.makedirs(log_dir, exist_ok=True)
#     file_path = os.path.join(log_dir, f"{agent_name}.log")

#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(level)
#     ch.setFormatter(logging.Formatter("%(message)s"))

#     fh = TimedRotatingFileHandler(
#         file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8"
#     )
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

# # ---------- JSON coercion helpers ----------

# def _looks_json_like(s: str) -> bool:
#     # quick heuristic to avoid trying loads on arbitrary text
#     s = s.strip()
#     return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

# def _try_parse_json(value: Any) -> Any:
#     """
#     If value is a str that looks like JSON, return json.loads(value).
#     Otherwise return value unchanged.
#     """
#     if isinstance(value, str) and _looks_json_like(value):
#         try:
#             return json.loads(value)
#         except Exception:
#             return value
#     return value

# def _deep_jsonify_toolcall_args(args: Any) -> Any:
#     """
#     Tool-call args can arrive as a JSON string or a dict already.
#     Make them JSON-objects if possible.
#     """
#     if args is None:
#         return None
#     if isinstance(args, str):
#         return _try_parse_json(args)
#     # Sometimes tool frameworks pass pydantic models or attrs classes:
#     try:
#         # Pydantic v2 Model
#         if hasattr(args, "model_dump"):
#             return args.model_dump()
#         # Pydantic v1 Model
#         if hasattr(args, "dict"):
#             return args.dict()
#     except Exception:
#         pass
#     return args

# # ---------- Emit helper ----------

# def _emit(
#     logger: logging.Logger,
#     event: str,
#     agent_name: str,
#     data: Any,
#     ts: Optional[str] = None,
#     pretty_json: bool = False,
# ) -> None:
#     obj = {
#         "event": event,
#         "agent": agent_name,
#         "ts": ts or (datetime.now().isoformat(timespec="seconds")),
#         "data": data,
#     }
#     if pretty_json:
#         logger.info(json.dumps(obj, ensure_ascii=False, indent=2))
#     else:
#         logger.info(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

# # ---------- Public API: unified activity logger ----------

# def log_tool_activity(
#     messages: Sequence[Any],
#     ai_msg: Optional[Any] = None,
#     agent_name: str = "agent",
#     logger: Optional[logging.Logger] = None,
#     header: str = "Tool Activity",
#     pretty_json: bool = False,
# ) -> None:
#     """
#     Logs planned tool calls from the assistant message (ai_msg) and
#     recent tool results found at the tail of `messages`.

#     If tool args/results were serialized as strings (with \n, \" etc.),
#     they are parsed into real JSON objects for clean logging.
#     """
#     log = logger or get_tool_logger(agent_name)
#     now = datetime.now().isoformat(timespec="seconds")

#     # Planned tool calls
#     tool_calls = getattr(ai_msg, "tool_calls", None)
#     if tool_calls:
#         _emit(log, "banner", agent_name, f"{header}: planned", ts=now, pretty_json=pretty_json)

#         for tc in tool_calls or []:
#             try:
#                 if isinstance(tc, dict):
#                     name = tc.get("name")
#                     args = tc.get("args")
#                 else:
#                     name = getattr(tc, "name", None)
#                     args = getattr(tc, "args", None)
#             except Exception:
#                 name, args = str(tc), None

#             args = _deep_jsonify_toolcall_args(args)
#             _emit(
#                 log,
#                 "tool_call_planned",
#                 agent_name,
#                 {"name": name, "args": args},
#                 ts=now,
#                 pretty_json=pretty_json,
#             )

#         _emit(log, "banner_end", agent_name, "end", ts=now, pretty_json=pretty_json)

#     # Recent tool results (walk back through consecutive tool messages)
#     i = len(messages) - 1
#     printed_results = False
#     while i >= 0 and _msg_type(messages[i]) == "tool":
#         if not printed_results:
#             _emit(log, "banner", agent_name, f"{header}: results", ts=now, pretty_json=pretty_json)
#             printed_results = True

#         tm = messages[i]
#         content = _msg_content(tm)

#         # Many tool frameworks put the result as a JSON string; parse if possible.
#         content = _try_parse_json(content)

#         _emit(
#             log,
#             "tool_result",
#             agent_name,
#             {"tool_call_id": _tool_call_id(tm), "result": content},
#             ts=now,
#             pretty_json=pretty_json,
#         )
#         i -= 1

#     if printed_results:
#         _emit(log, "banner_end", agent_name, "end", ts=now, pretty_json=pretty_json)


# test 2

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
# --- add helpers ---
def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)  # if it's JSON, parse it
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: _maybe_parse_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_maybe_parse_json(v) for v in value]
    return value

# --- add this helper somewhere above _emit ---
def _json_sanitize(x):
    # primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    # common containers
    if isinstance(x, set):
        # keep deterministic ordering for nicer diffs/logs
        try:
            return sorted(x)
        except Exception:
            return list(x)
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    # datetimes
    try:
        from datetime import datetime, date
        if isinstance(x, (datetime, date)):
            return x.isoformat()
    except Exception:
        pass
    # pydantic v2 / v1 models
    if hasattr(x, "model_dump"):
        try:
            return _json_sanitize(x.model_dump())
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return _json_sanitize(x.dict())
        except Exception:
            pass
    # generic objects
    if hasattr(x, "__dict__"):
        return _json_sanitize(vars(x))
    # final fallback
    return str(x)


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
        "ts": ts or (datetime.now().isoformat(timespec="seconds")),
        "data": _json_sanitize(data),  # ← sanitize everything here
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
    now = datetime.now().isoformat(timespec="seconds")

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