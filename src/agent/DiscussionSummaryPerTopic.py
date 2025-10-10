# =============================================================================
# Module: discussion_summary_per_topic_agent
# =============================================================================
# Overview
#   Production-grade LangGraph agent that generates a per-topic discussion
#   summary for each interview topic, in parallel. It runs a compact ReAct-style
#   inner loop with Mongo-backed tools and coerces the final assistant content
#   into the typed schema: DiscussionSummaryPerTopicSchema.DiscussionTopic.
#
# Responsibilities
#   - Build a System prompt per topic from the global generated summary + topic.
#   - Invoke an LLM bound to Mongo tools (via ToolNode) for each topic.
#   - Convert the final tool-free assistant message to the typed schema.
#   - Run all topics concurrently, preserving the input topic names exactly.
#   - Retry when the set of produced topics doesn’t match the input set.
#
# Data Flow
#   Outer Graph:
#     START ──► discussion_summary_per_topic_generator
#                 ├─► END (should_regenerate=False)
#                 └─► discussion_summary_per_topic_generator (should_regenerate=True)
#
#   Inner Graph (per topic):
#     agent ─► (tools)* ─► respond
#       1) agent: tool-enabled LLM plans tools if needed
#       2) tools: ToolNode executes planned tool calls
#       3) respond: coerce last tool-free assistant message to schema
#
# Reliability & Observability
#   - Timeouts + retries (exponential backoff) for LLM and tools.
#   - Console + rotating file logs with compact redaction for large fields.
#
# Configuration (Environment Variables)
#   DISCUSSION_SUMMARY_AGENT_LOG_DIR, DISCUSSION_SUMMARY_AGENT_LOG_FILE,
#   DISCUSSION_SUMMARY_AGENT_LOG_LEVEL, DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_WHEN,
#   DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_INTERVAL, DISCUSSION_SUMMARY_AGENT_LOG_BACKUP_COUNT,
#   DISC_AGENT_LLM_TIMEOUT_SECONDS, DISC_AGENT_LLM_RETRIES, DISC_AGENT_LLM_RETRY_BACKOFF_SECONDS,
#   DISC_AGENT_TOOL_TIMEOUT_SECONDS, DISC_AGENT_TOOL_RETRIES, DISC_AGENT_TOOL_RETRY_BACKOFF_SECONDS,
#   DISC_AGENT_TOOL_MAX_WORKERS
# =============================================================================

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from logging.handlers import TimedRotatingFileHandler
from string import Template
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import PrivateAttr

from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import DiscussionSummaryPerTopicSchema
from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
    DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT,
)
from ..model_handling import llm_dts


# ==============================
# Config (env-overridable)
# ==============================

AGENT_NAME = "discussion_summary_agent"

LOG_DIR = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_DIR", "logs")
LOG_LEVEL = getattr(
    logging, os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO
)
LOG_FILE = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_BACKUP_COUNT", "365"))

# Retry/timeout knobs (namespaced for this agent)
LLM_TIMEOUT_SECONDS: float = float(os.getenv("DISC_AGENT_LLM_TIMEOUT_SECONDS", "90"))
LLM_RETRIES: int = int(os.getenv("DISC_AGENT_LLM_RETRIES", "2"))
LLM_BACKOFF_SECONDS: float = float(os.getenv("DISC_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

# --- richer preview knobs for 'question_guidelines' ---
SHOW_FULL_TEXT = os.getenv("QA_LOG_SHOW_FULL_TEXT", "0") == "1"
SHOW_FULL_FIELDS = {
    k.strip().lower()
    for k in os.getenv("QA_LOG_SHOW_FULL_FIELDS", "").split(",")
    if k.strip()
}
QGUIDE_PREVIEW_LEN = int(os.getenv("QA_LOG_QGUIDE_PREVIEW_LEN", "280"))      # chars
QGUIDE_PREVIEW_LINES = int(os.getenv("QA_LOG_QGUIDE_PREVIEW_LINES", "2"))    # lines

TOOL_TIMEOUT_SECONDS: float = float(os.getenv("DISC_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
TOOL_RETRIES: int = int(os.getenv("DISC_AGENT_TOOL_RETRIES", "2"))
TOOL_BACKOFF_SECONDS: float = float(os.getenv("DISC_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))
# Tool payload logging: 'off' | 'summary' | 'full'
TOOL_LOG_PAYLOAD = os.getenv("DISC_AGENT_TOOL_LOG_PAYLOAD", "summary").strip().lower()
# valid: off, summary, full

_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("DISC_AGENT_TOOL_MAX_WORKERS", "8")))

# ---------- clean log redaction config ----------
RAW_TEXT_FIELDS = {
    "question_guidelines", "guidelines", "template", "prompt",
    "policy", "notes", "rubric", "examples", "description_md",
}

GUIDELINE_KEY_CANDIDATES = {
    "question_guidelines", "question_guideline", "guidelines", "guideline"
}

# Global retry counter used only for logging iteration counts
disc_retry_counter = 1


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
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, filename)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    rotating_file = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8", utc=False, delay=True
    )
    logging.raiseExceptions = False  # production: don’t raise on logging I/O errors
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


def _is_guideline_key(key: str) -> bool:
    """True for any likely guidelines field (case-insensitive)."""
    k = (key or "").lower()
    if k in GUIDELINE_KEY_CANDIDATES:
        return True
    # also catch variations like "global_guidelines", "question_guidelines_md"
    return "guideline" in k


def _redact(value: Any, *, omit_fields: bool, preview_len: int = 140) -> Any:
    """
    Redact long raw text fields for compact logging.
    Always include a compact preview for any *guideline-like* key, not just 'question_guidelines'.
    """
    value = _jsonish(value)

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key_l = k.lower() if isinstance(k, str) else k

            if isinstance(k, str) and key_l in RAW_TEXT_FIELDS and isinstance(v, str):
                # show full only if explicitly allowed
                if SHOW_FULL_TEXT or (SHOW_FULL_FIELDS and key_l in SHOW_FULL_FIELDS):
                    out[k] = v
                    continue

                # guidelines preview (broadened)
                if _is_guideline_key(key_l):
                    lines = [ln.strip() for ln in v.strip().splitlines() if ln.strip()]
                    head = " ".join(lines[:max(1, QGUIDE_PREVIEW_LINES)]) or ""
                    if len(head) > QGUIDE_PREVIEW_LEN:
                        head = head[:QGUIDE_PREVIEW_LEN].rstrip() + "…"
                    out[f"{k}_preview"] = head
                    out[f"{k}_len"] = len(v)
                    continue

                # other long raw fields
                if omit_fields:
                    continue
                head = (v.strip().splitlines() or [""])[0]
                if len(head) > preview_len:
                    head = head[:preview_len].rstrip() + "…"
                out[f"{k}_preview"] = head
                out[f"{k}_len"] = len(v)

            else:
                out[k] = _redact(v, omit_fields=omit_fields, preview_len=preview_len)
        return out

    if isinstance(value, (list, tuple)):
        return [_redact(v, omit_fields=omit_fields, preview_len=preview_len) for v in value]

    # primitives (incl. plain strings) pass through unchanged
    return value


def _summarize_payload(payload: Any) -> str:
    """One-liner without dumping big JSON."""
    try:
        obj = _jsonish(payload)
        if isinstance(obj, dict):
            keys = list(obj.keys())
            preview = keys[:6]
            extra = len(keys) - len(preview)
            extra_txt = f"+{extra} more" if extra > 0 else ""
            ok = obj.get("ok")
            cnt = obj.get("count")
            has_data = "data" in obj
            return f"dict(keys={preview}{', ' + extra_txt if extra_txt else ''}; ok={ok}; count={cnt}; data={'yes' if has_data else 'no'})"
        if isinstance(obj, list):
            return f"list(len={len(obj)})"
        return type(obj).__name__
    except Exception:
        return "<unavailable>"


def _gated_payload_str(payload: Any, *, omit_fields: bool = True) -> str:
    """
    off    -> '<hidden>'
    summary-> compact one-liner
    full   -> pretty/redacted JSON (respects omit_fields)
    """
    mode = TOOL_LOG_PAYLOAD
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        return _summarize_payload(payload)
    compact = _redact(_jsonish(payload), omit_fields=omit_fields)
    return _compact(compact)


def log_json(label: str, payload: Any, level: int = logging.INFO, omit_fields: bool = True) -> None:
    logger.log(level, f"{label}: {_gated_payload_str(payload, omit_fields=omit_fields)}")


def log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    """Log planned tool calls and trailing tool results, honoring TOOL_LOG_PAYLOAD."""
    if not messages:
        return

    planned = getattr(ai_msg, "tool_calls", None)
    if planned:
        log_info("Tool plan:")
        for tc in planned:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            logger.info(f"  planned -> {name} args={_gated_payload_str(args, omit_fields=False)}")

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
        logger.info(f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_gated_payload_str(content, omit_fields=True)}")


def log_retry_iteration(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    """Human format output."""
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
    # noinspection PyUnboundLocalVariable
    raise last_exc


# ======================================================
# Tool wrapper with timeout + retry
# ======================================================

class RetryTool(BaseTool):
    """Wrap BaseTool to add timeout + retry (sync/async)."""

    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff: float = PrivateAttr()

    def __init__(self, inner: BaseTool, *, retries: int, timeout_s: float, backoff_base_s: float) -> None:
        name = getattr(inner, "name", inner.__class__.__name__)
        description = getattr(inner, "description", "") or "Retried tool wrapper"
        args_schema = getattr(inner, "args_schema", None)
        super().__init__(name=name, description=description, args_schema=args_schema)
        self._inner = inner
        self._retries = retries
        self._timeout_s = timeout_s
        self._backoff = backoff_base_s

    def _run(self, *args, **kwargs):
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
                last_exc = exc
                log_retry_iteration(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                last_exc = exc
                log_retry_iteration(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff * (2 ** (attempt - 1)))
        # noinspection PyUnboundLocalVariable
        raise last_exc

    async def _arun(self, *args, **kwargs):
        config = kwargs.pop("config", None)

        async def _call_once():
            if hasattr(self._inner, "_arun"):
                return await self._inner._arun(*args, **{**kwargs, "config": config})
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._inner._run(*args, **{**kwargs, "config": config}))

        return await _retry_async(
            _call_once,
            retries=self._retries,
            timeout_s=self._timeout_s,
            backoff_base_s=self._backoff,
            retry_reason=f"tool_async:{self.name}",
        )


# ---------- Inner ReAct state for per-topic Mongo loop ----------
class _PerTopicState(MessagesState):
    final_response: DiscussionSummaryPerTopicSchema.DiscussionTopic


class PerTopicDiscussionSummaryGenerationAgent:
    """Generates a DiscussionTopic for a single input topic via an inner ReAct loop."""

    llm = llm_dts

    # wrap Mongo tools with retry/timeout
    _RAW_MONGO_TOOLS = get_mongo_tools(llm=llm)
    MONGO_TOOLS = [
        RetryTool(t, retries=TOOL_RETRIES, timeout_s=TOOL_TIMEOUT_SECONDS, backoff_base_s=TOOL_BACKOFF_SECONDS)
        for t in _RAW_MONGO_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(
        DiscussionSummaryPerTopicSchema.DiscussionTopic, method="function_calling"
    )

    _compiled_graph = None  # cache for inner graph

    # ---------- LLM helpers ----------
    @staticmethod
    async def _invoke_agent(messages: Sequence[Any]) -> Any:
        async def _call():
            if hasattr(PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.invoke, messages
            )

        log_info("Calling LLM (agent)")
        ai = await _retry_async(
            _call,
            retries=LLM_RETRIES,
            timeout_s=LLM_TIMEOUT_SECONDS,
            backoff_base_s=LLM_BACKOFF_SECONDS,
            retry_reason="llm:agent",
        )
        log_info("LLM (agent) call succeeded")
        return ai

    @staticmethod
    async def _invoke_structured(ai_content: str) -> DiscussionSummaryPerTopicSchema.DiscussionTopic:
        payload = [HumanMessage(content=ai_content)]

        async def _call():
            if hasattr(PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.invoke, payload
            )

        log_info("Calling LLM (structured)")
        obj = await _retry_async(
            _call,
            retries=LLM_RETRIES,
            timeout_s=LLM_TIMEOUT_SECONDS,
            backoff_base_s=LLM_BACKOFF_SECONDS,
            retry_reason="llm:structured",
        )
        log_info("LLM (structured) call succeeded")
        return obj

    # ---------- ASYNC NODE IMPLS (called by wrappers below) ----------
    @staticmethod
    async def _agent_node(state: _PerTopicState):
        log_tool_activity(state["messages"], ai_msg=None)
        ai = await PerTopicDiscussionSummaryGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _PerTopicState):
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

        final_obj = await PerTopicDiscussionSummaryGenerationAgent._invoke_structured(ai_content)
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _PerTopicState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ---------- RunnableLambda top-level async wrappers ----------
    # (avoids any staticmethod/class-binding ambiguity)
    @staticmethod
    async def _agent_node_async(state: _PerTopicState):
        return await PerTopicDiscussionSummaryGenerationAgent._agent_node(state)

    @staticmethod
    async def _respond_node_async(state: _PerTopicState):
        return await PerTopicDiscussionSummaryGenerationAgent._respond_node(state)

    # ---------- Compile inner ReAct graph ----------
    @classmethod
    def _get_graph(cls):
        if cls._compiled_graph is not None:
            return cls._compiled_graph

        workflow = StateGraph(_PerTopicState)
        workflow.add_node("agent", RunnableLambda(cls._agent_node_async))
        workflow.add_node("respond", RunnableLambda(cls._respond_node_async))
        workflow.add_node("tools", ToolNode(cls.MONGO_TOOLS, tags=["mongo-tools"]))
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

    @staticmethod
    async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any], thread_id: str):
        class AtTemplate(Template):
            delimiter = "@"

        tpl = AtTemplate(DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT)
        content = tpl.substitute(
            generated_summary=generated_summary_json,
            interview_topic=json.dumps(topic, ensure_ascii=False),
            thread_id=thread_id,
        )

        sys_message = SystemMessage(content=content)
        trigger_message = HumanMessage(content="Based on the provided instructions please start the process")

        graph = PerTopicDiscussionSummaryGenerationAgent._get_graph()
        result = await graph.ainvoke({"messages": [sys_message, trigger_message]})
        return result["final_response"]

    # ---------- Regeneration policy ----------
    @staticmethod
    async def should_regenerate(state: AgentInternalState):
        """
        Regenerate if the set of topics in the output does not exactly match
        the set of input topics. Includes a guard to avoid infinite loops if
        nothing was produced (single extra retry).
        """
        global disc_retry_counter

        # Guard: if nothing produced, allow at most 1 retry under this condition
        if not getattr(state, "discussion_summary_per_topic", None) or \
           not getattr(state.discussion_summary_per_topic, "discussion_topics", None):
            log_retry_iteration("No discussion topics produced; retrying once", disc_retry_counter)
            disc_retry_counter += 1
            return disc_retry_counter <= 2  # retry only once for the "no output" case

        input_topics = {t.topic for t in state.interview_topics.interview_topics}
        output_topics = {dt.topic for dt in state.discussion_summary_per_topic.discussion_topics}
        if input_topics != output_topics:
            missing = input_topics - output_topics
            extra = output_topics - input_topics
            log_retry_iteration(
                reason="Topic mismatch",
                iteration=disc_retry_counter,
                extra={"missing": missing, "extra": extra},
            )
            disc_retry_counter += 1
            return True
        return False

    # ---------- Generation graph node ----------
    @staticmethod
    async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Generate summaries for all topics concurrently:
          1) Normalize the input topics
          2) Serialize generated_summary for prompting
          3) Launch one inner-graph run per topic via asyncio.gather
          4) Force output topic names to match input names exactly
        """
        # Normalize topics list coming from parent state
        try:
            topics_list: List[Dict[str, Any]] = [t.model_dump() for t in state.interview_topics.interview_topics]
        except Exception:
            topics_list = state.interview_topics  # already a list[dict]

        if not isinstance(topics_list, list) or len(topics_list) == 0:
            raise ValueError("interview_topics must be a non-empty list[dict]")

        # Serialize generated_summary for prompt
        try:
            generated_summary_json = state.generated_summary.model_dump_json()
        except Exception:
            generated_summary_json = json.dumps(state.generated_summary, ensure_ascii=False)

        # Run all topic calls concurrently (each via inner graph)
        tasks = [
            asyncio.create_task(
                PerTopicDiscussionSummaryGenerationAgent._one_topic_call(
                    generated_summary_json, topic, state.id
                )
            )
            for topic in topics_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful DiscussionTopic entries and enforce exact topic names
        discussion_topics: List[Any] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                log_warning(f"Topic {idx} summarization failed: {result}")
                continue

            input_topic_name = (
                topics_list[idx].get("topic")
                or topics_list[idx].get("name")
                or topics_list[idx].get("title")
                or "Unknown"
            )
            try:
                if hasattr(result, "model_copy"):  # pydantic v2
                    result = result.model_copy(update={"topic": input_topic_name})
                elif hasattr(result, "copy"):  # pydantic v1
                    result = result.copy(update={"topic": input_topic_name})
                elif isinstance(result, dict):
                    result["topic"] = input_topic_name
                else:
                    setattr(result, "topic", input_topic_name)
                discussion_topics.append(result)
            except Exception as exc:
                log_warning(f"Failed to append structured response for topic index {idx}: {exc}")

        state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
            discussion_topics=discussion_topics
        )
        log_info("Per-topic discussion summaries generated")
        return state

    # ----------  Topic wise discussion summary graph ----------
    @staticmethod
    def get_graph(checkpointer=None):
        """
        Topic wise discussion summary graph:
        START -> discussion_summary_per_topic_generator
               -> (should_regenerate ? discussion_summary_per_topic_generator : END)
        """
        gb = StateGraph(AgentInternalState)
        gb.add_node(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.discussion_summary_per_topic_generator,
        )
        gb.add_edge(START, "discussion_summary_per_topic_generator")
        gb.add_conditional_edges(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.should_regenerate,
            {False: END, True: "discussion_summary_per_topic_generator"},
        )
        gb.add_edge("discussion_summary_per_topic_generator", END)
        return gb.compile(checkpointer=checkpointer, name="PerTopicDiscussionSummaryGenerationAgent")


if __name__ == "__main__":
    graph = PerTopicDiscussionSummaryGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
