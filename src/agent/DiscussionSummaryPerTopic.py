import json
import asyncio
import logging
import os
import sys
import time
from string import Template
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from logging.handlers import TimedRotatingFileHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda
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
    logging,
    os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)
LOG_FILE = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_BACKUP_COUNT", "365"))

# Retry/timeout knobs (namespaced for this agent)
LLM_TIMEOUT_SECONDS: float = float(os.getenv("DISC_AGENT_LLM_TIMEOUT_SECONDS", "90"))
LLM_RETRIES: int = int(os.getenv("DISC_AGENT_LLM_RETRIES", "2"))
LLM_BACKOFF_SECONDS: float = float(os.getenv("DISC_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

TOOL_TIMEOUT_SECONDS: float = float(os.getenv("DISC_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
TOOL_RETRIES: int = int(os.getenv("DISC_AGENT_TOOL_RETRIES", "2"))
TOOL_BACKOFF_SECONDS: float = float(os.getenv("DISC_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))

_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("DISC_AGENT_TOOL_MAX_WORKERS", "8")))

# ---------- clean log redaction config ----------
RAW_TEXT_FIELDS = {
    "question_guidelines", "guidelines", "template", "prompt",
    "policy", "notes", "rubric", "examples", "description_md",
}

# Global retry counter used only for logging iteration counts
count = 1

# ==============================
# Human-style logging
# ==============================


def _build_logger(
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

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Rotating file handler
    fh = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8", utc=False
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


LOGGER = _build_logger(
    name=AGENT_NAME,
    log_dir=LOG_DIR,
    level=LOG_LEVEL,
    filename=LOG_FILE,
    when=LOG_ROTATE_WHEN,
    interval=LOG_ROTATE_INTERVAL,
    backup_count=LOG_BACKUP_COUNT,
)


def log_info(msg: str) -> None:
    LOGGER.info(msg)


def log_warning(msg: str) -> None:
    LOGGER.warning(msg)


def log_error(msg: str) -> None:
    LOGGER.error(msg)


# ---------- tiny utils ----------
def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def _jsonish(x: Any) -> Any:
    if isinstance(x, str) and _looks_like_json(x):
        try:
            return json.loads(x)
        except Exception:
            return x
    if isinstance(x, dict):
        return {k: _jsonish(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonish(v) for v in x]
    return x


def _compact(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, indent=2) if isinstance(x, (dict, list)) else str(x)
    except Exception:
        return str(x)


def _walk(o: Any, path: Tuple[Any, ...] = ()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, path + (k,))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _walk(v, path + (i,))
    else:
        yield path, o


# ---------- redaction for compact logs ----------
def _redact(o: Any, *, omit_fields: bool, preview_len: int = 140) -> Any:
    o = _jsonish(o)
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            key = k.lower() if isinstance(k, str) else k
            if isinstance(k, str) and key in RAW_TEXT_FIELDS and isinstance(v, str):
                if omit_fields:
                    continue
                head = (v.strip().splitlines() or [""])[0]
                head = head[:preview_len] + ("…" if len(head) > preview_len else "")
                out[k] = f"<{k}: {len(v)} chars; see raw block below — \"{head}\">"
            else:
                out[k] = _redact(v, omit_fields=omit_fields, preview_len=preview_len)
        return out
    if isinstance(o, (list, tuple)):
        return [_redact(v, omit_fields=omit_fields, preview_len=preview_len) for v in o]
    return o


# ---------- main entry ----------
def log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    if not messages:
        return

    # Planned tool calls
    tcalls = getattr(ai_msg, "tool_calls", None)
    if tcalls:
        log_info("Tool plan:")
        for tc in tcalls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            LOGGER.info(f"  planned -> {name} args={_compact(_redact(_jsonish(args), omit_fields=False))}")

    # Trailing tool results
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
        LOGGER.info(f"  result <- id={getattr(tm, 'tool_call_id', None)} data={_compact(compact)}")


def log_retry_iteration(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    """Human format output."""
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


# ======================================
# Retry / timeout helper for async ops
# ======================================

async def _retry_async(
    op_factory,
    *,
    retries: int,
    timeout_s: float,
    backoff_base_s: float,
    retry_reason: str,
    iteration_start: int = 1,
):
    attempt = 0
    last_exc = None
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
    raise last_exc


# ======================================================
# Tool wrapper with timeout + retry
# ======================================================

class RetryTool(BaseTool):
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
        last_exc = None
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
    llm_dts = llm_dts

    # wrap Mongo tools with retry/timeout
    _RAW_MONGO_TOOLS = get_mongo_tools(llm=llm_dts)
    MONGO_TOOLS = [
        RetryTool(t, retries=TOOL_RETRIES, timeout_s=TOOL_TIMEOUT_SECONDS, backoff_base_s=TOOL_BACKOFF_SECONDS)
        for t in _RAW_MONGO_TOOLS
    ]

    _AGENT_MODEL = llm_dts.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_dts.with_structured_output(
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
            return await loop.run_in_executor(None, PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.invoke, messages)

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
            return await loop.run_in_executor(None, PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.invoke, payload)

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

    # ---------- Lazy compile inner ReAct graph ----------
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
    async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any], thread_id):
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
        nothing was produced.
        """
        global count

        # Guard: if nothing produced, allow at most 1 retry in this condition
        if not getattr(state, "discussion_summary_per_topic", None) or \
           not getattr(state.discussion_summary_per_topic, "discussion_topics", None):
            log_retry_iteration("No discussion topics produced; retrying once", count)
            count += 1
            # retry only once under this specific condition
            return count <= 2

        input_topics = {t.topic for t in state.interview_topics.interview_topics}
        output_topics = {dt.topic for dt in state.discussion_summary_per_topic.discussion_topics}
        if input_topics != output_topics:
            missing = input_topics - output_topics
            extra = output_topics - input_topics
            log_retry_iteration(
                reason="Topic mismatch",
                iteration=count,
                extra={"missing": missing, "extra": extra},
            )
            count += 1
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
        discussion_topics = []
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                log_warning(f"Topic {idx} summarization failed: {r}")
                continue

            in_topic = (
                topics_list[idx].get("topic")
                or topics_list[idx].get("name")
                or topics_list[idx].get("title")
                or "Unknown"
            )
            try:
                if hasattr(r, "model_copy"):  # pydantic v2
                    r = r.model_copy(update={"topic": in_topic})
                elif hasattr(r, "copy"):  # pydantic v1
                    r = r.copy(update={"topic": in_topic})
                elif isinstance(r, dict):
                    r["topic"] = in_topic
                else:
                    setattr(r, "topic", in_topic)
                discussion_topics.append(r)
            except Exception as e:
                log_warning(f"Failed to append structured response for topic index {idx}: {e}")

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
