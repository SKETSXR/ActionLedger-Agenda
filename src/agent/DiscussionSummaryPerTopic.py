import json
import asyncio
import logging
import os
import sys
from string import Template
from typing import Any, Dict, List, Optional, Sequence, Tuple

from logging.handlers import TimedRotatingFileHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage

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


# ---------- config ----------
RAW_TEXT_FIELDS = {
    "question_guidelines", "guidelines", "template", "prompt",
    "policy", "notes", "rubric", "examples", "description_md",
}


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

    # 1) Compact (omit long-text fields) for all results
    for tm in tool_msgs:
        content = getattr(tm, "content", None)
        compact = _redact(_jsonish(content), omit_fields=True)
        LOGGER.info(f"  result <- id={getattr(tm, 'tool_call_id', None)} data={_compact(compact)}")


def log_retry_iteration(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    """Human format output."""
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


# ---------- Inner ReAct state for per-topic Mongo loop ----------
class _PerTopicState(MessagesState):
    """State container for the inner ReAct-style loop that summarizes a single topic."""
    final_response: DiscussionSummaryPerTopicSchema.DiscussionTopic


class PerTopicDiscussionSummaryGenerationAgent:
    """
    Generates per-topic discussion summaries using a small tool-using ReAct loop.
    The loop can call Mongo tools, and the final assistant output is coerced
    into DiscussionSummaryPerTopicSchema.DiscussionTopic.
    """

    # LLM & tools (class attributes so clients/bindings are reused)
    llm_dts = llm_dts
    MONGO_TOOLS = get_mongo_tools(llm=llm_dts)
    _AGENT_MODEL = llm_dts.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_dts.with_structured_output(
        DiscussionSummaryPerTopicSchema.DiscussionTopic, method="function_calling"
    )

    # ---------- Inner graph (agent -> tools -> agent ... -> respond) ----------
    @staticmethod
    def _agent_node(state: _PerTopicState):
        """
        Invoke the tool-enabled model. If the assistant plans tool calls,
        LangGraph will execute them via the ToolNode.
        """
        log_info("Calling LLM (agent)")
        log_tool_activity(state["messages"], ai_msg=None)
        ai = PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        log_info("LLM (agent) call succeeded")
        return {"messages": [ai]}

    @staticmethod
    def _respond_node(state: _PerTopicState):
        """
        Take the most recent assistant message without tool calls and coerce it
        into DiscussionTopic via the structured-output model. If none exists,
        fall back to the most recent assistant message, then to the last message.
        """
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

        log_info("Calling LLM (structured)")
        final_obj = PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.invoke(
            [HumanMessage(content=ai_content)]
        )
        log_info("LLM (structured) call succeeded")
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _PerTopicState):
        """
        Router: if the latest assistant message contains tool calls,
        continue to the ToolNode, otherwise, proceed to respond.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # Compile the inner ReAct graph once
    _workflow = StateGraph(_PerTopicState)
    _workflow.add_node("agent", _agent_node)
    _workflow.add_node("respond", _respond_node)
    _workflow.add_node("tools", ToolNode(MONGO_TOOLS, tags=["mongo-tools"]))
    _workflow.set_entry_point("agent")
    _workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"continue": "tools", "respond": "respond"},
    )
    _workflow.add_edge("tools", "agent")
    _workflow.add_edge("respond", END)
    _per_topic_graph = _workflow.compile()

    # ---------- Single-topic runner ----------
    @staticmethod
    async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any], thread_id):
        """
        Runs the inner graph for a single topic:
        System prompt -> agent (may call Mongo tools) -> respond (structured DiscussionTopic).
        """
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

        result = await PerTopicDiscussionSummaryGenerationAgent._per_topic_graph.ainvoke(
            {"messages": [sys_message, trigger_message]}
        )
        return result["final_response"]

    # ---------- Regeneration policy ----------
    @staticmethod
    async def should_regenerate(state: AgentInternalState):
        """
        Regenerate if the set of topics in the output does not exactly match
        the set of input topics.
        """
        global count

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
                # keep behavior: skip failed topic silently, but log it
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
