import json
import copy
import re
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import List, Any, Dict, Optional, Tuple, Sequence

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from string import Template

from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import QASetsSchema
from ..prompt.qa_agent_prompt import QA_BLOCK_AGENT_PROMPT
from ..model_handling import llm_qa


# ==============================
# Config (env-overridable)
# ==============================

AGENT_NAME = os.getenv("QA_AGENT_NAME", "qa_block_generation_agent")

LOG_DIR = os.getenv("QA_AGENT_LOG_DIR", "logs")
LOG_LEVEL = getattr(logging, os.getenv("QA_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FILE = os.getenv("QA_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("QA_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("QA_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("QA_AGENT_LOG_BACKUP_COUNT", "365"))

# Global retry counter used only for logging iteration counts.
count = 1
SHOW_FULL_TEXT = os.getenv("QA_LOG_SHOW_FULL_TEXT", "0") == "1"
# Optionally restrict to specific keys (comma-separated), e.g. "question_guidelines"
SHOW_FULL_FIELDS = {
    k.strip().lower()
    for k in os.getenv("QA_LOG_SHOW_FULL_FIELDS", "").split(",")
    if k.strip()
}


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
                # If we're omitting long text fields entirely
                if omit_fields:
                    continue

                # If we are allowed to show full text (globally or for selected fields)
                if SHOW_FULL_TEXT or (SHOW_FULL_FIELDS and key in SHOW_FULL_FIELDS):
                    out[k] = v  # print the raw, full text
                else:
                    # fallback: short, clean preview (no angle brackets, no "see raw block below")
                    head = (v.strip().splitlines() or [""])[0]
                    if len(head) > preview_len:
                        head = head[:preview_len].rstrip() + "…"
                    out[k + "_preview"] = head
                    out[k + "_len"] = len(v)
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
        # use preview, not omission
        compact = _redact(_jsonish(content), omit_fields=False)
        LOGGER.info(f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_compact(compact)}")


def log_retry_iteration(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


# ---------- Inner ReAct state for per-topic QA generation ----------
class _QAInnerState(MessagesState):
    """State container for the inner ReAct loop that generates QA blocks."""
    final_response: QASetsSchema


class QABlockGenerationAgent:
    """
    Generates structured QA blocks for each topic's deep-dive nodes by running a
    tool-using inner ReAct loop (with Mongo tools), coercing the final assistant
    content into the QASetsSchema, and validating basic constraints.
    """

    # LLM & tools (kept as class attributes to reuse model clients and tool bindings)
    llm_qa = llm_qa
    MONGO_TOOLS = get_mongo_tools(llm=llm_qa)
    _AGENT_MODEL = llm_qa.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_qa.with_structured_output(QASetsSchema, method="function_calling")

    # ---------- Inner graph nodes ----------
    @staticmethod
    def _agent_node(state: _QAInnerState):
        """
        Invoke the tool-enabled model. If the assistant plans tool calls,
        LangGraph will execute them via the ToolNode.
        """
        log_info("Calling LLM (agent)")
        log_tool_activity(messages=state["messages"], ai_msg=None)
        ai = QABlockGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        log_info("LLM (agent) call succeeded")
        return {"messages": [ai]}

    @staticmethod
    def _respond_node(state: _QAInnerState):
        """
        Take the most recent assistant message without tool calls and coerce it
        into QASetsSchema using the structured-output model. If none exists,
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
        final_obj = QABlockGenerationAgent._STRUCTURED_MODEL.invoke([HumanMessage(content=ai_content)])
        log_info("LLM (structured) call succeeded")
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _QAInnerState):
        """
        Router node: if the latest assistant message contains tool calls,
        continue to the ToolNode; otherwise, proceed to respond.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(messages=state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ---------- Compile inner ReAct graph once ----------
    _workflow = StateGraph(_QAInnerState)
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
    _qa_inner_graph = _workflow.compile()

    # ----------------- utilities -----------------
    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        """Best-effort conversion to a dict without altering semantics."""
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "dict"):
            return x.dict()
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _get_topic_name(obj: Any) -> str:
        """Extract a human-friendly topic name from common keys; default to 'Unknown'."""
        d = QABlockGenerationAgent._as_dict(obj)
        for k in ("topic", "name", "title", "label"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    @staticmethod
    def _can(s: str) -> str:
        """Canonicalize a string for lookup (lowercase, collapse spaces, remove punctuation)."""
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return s

    @staticmethod
    def _extract_topic_name_from_summary(summary_obj: Any) -> str:
        """Extract the topic name from known summary keys; default to 'Unknown'."""
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
        Generate a single QA set for a topic. The number of QA blocks must match
        the number of deep-dive nodes. Performs strict post-generation checks:
          - block count equals deep-dive count
          - each block has exactly 7 qa_items
          - no Easy difficulty for Counter Question items
        Returns: (qa_set_dict, error_message) where error_message == "" if OK.
        """
        class AtTemplate(Template):
            delimiter = "@"

        deep_dive_nodes = json.loads(deep_dive_nodes_json or "[]")
        n_blocks = len(deep_dive_nodes)

        COUNT_DIRECTIVE = f"""
        IMPORTANT COUNT RULE:
        - This topic has {n_blocks} deep-dive nodes. Output exactly {n_blocks} QA blocks (one per deep-dive node, in order).
        """

        tpl = AtTemplate(QA_BLOCK_AGENT_PROMPT + COUNT_DIRECTIVE)
        sys_content = tpl.substitute(
            discussion_summary=discussion_summary_json,
            deep_dive_nodes=deep_dive_nodes_json,
            thread_id=thread_id,
            qa_error=qa_error or "",
        )

        sys_message = SystemMessage(content=sys_content)
        trigger_message = HumanMessage(content="Based on the provided instructions please start the process")

        try:
            # Run inner ReAct loop (tools + LLM), then coerce to QASetsSchema via _respond_node.
            result = await QABlockGenerationAgent._qa_inner_graph.ainvoke(
                {"messages": [sys_message, trigger_message]}
            )
            schema = result["final_response"]  # QASetsSchema

            obj = schema.model_dump() if hasattr(schema, "model_dump") else schema
            sets = obj.get("qa_sets", []) or []
            if not sets:
                return {"topic": topic_name, "qa_blocks": []}, "No qa_sets produced."

            one = sets[0]
            one["topic"] = topic_name
            blocks = one.get("qa_blocks", []) or []

            # Hard validations
            errs = []
            if len(blocks) != n_blocks:
                errs.append(f"Expected {n_blocks} blocks, got {len(blocks)}.")
            for i, b in enumerate(blocks, start=1):
                qi = b.get("qa_items", []) or []
                if len(qi) != 7:
                    errs.append(f"Block {i} must have 7 qa_items, got {len(qi)}.")
                for item in qi:
                    if (item.get("q_type") == "Counter Question" and item.get("q_difficulty") == "Easy"):
                        errs.append(f"Block {i} has an Easy counter (qa_id={item.get('qa_id')}); not allowed.")

            if errs:
                return one, " ; ".join(errs)

            return one, ""

        except (ValidationError, OutputParserException) as e:
            return {"topic": topic_name, "qa_blocks": []}, f"Parser/Schema error: {e}"
        except Exception as e:
            return {"topic": topic_name, "qa_blocks": []}, f"Generation error: {e}"

    @staticmethod
    async def qablock_generator(state: AgentInternalState) -> AgentInternalState:
        """
        For each topic in state.nodes.topics_with_nodes:
          - locate its discussion summary
          - collect its deep-dive nodes
          - run the inner ReAct loop to produce QA blocks
        Aggregates results into state.qa_blocks (QASetsSchema) or records errors.
        """
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")
        if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
            raise ValueError("nodes (topics_with_nodes) are required before QA block generation.")

        # Normalize summaries_list
        raw = state.discussion_summary_per_topic
        if hasattr(raw, "discussion_topics"):
            summaries_list = list(raw.discussion_topics)
        elif isinstance(raw, (list, tuple)):
            summaries_list = list(raw)
        elif isinstance(raw, dict) and "discussion_topics" in raw:
            summaries_list = list(raw["discussion_topics"])
        else:
            raise ValueError("discussion_summary_per_topic has no 'discussion_topics' field")

        final_sets: List[Dict[str, Any]] = []
        accumulated_errs: List[str] = []

        # Quick index by canonicalized topic name
        summaries_by_can: Dict[str, Any] = {}
        for s in summaries_list:
            nm = QABlockGenerationAgent._extract_topic_name_from_summary(s)
            summaries_by_can[QABlockGenerationAgent._can(nm)] = s

        # Aliases that count as "deep-dive" nodes
        DEEP_DIVE_ALIASES = {"deep dive", "deep_dive", "deep-dive", "probe", "follow up", "follow-up"}
        covered = set()

        log_info("QA block generation started")

        # -------- Pass A: generate QA blocks for topics with deep-dive nodes --------
        for topic_entry in state.nodes.topics_with_nodes:
            topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else dict(topic_entry)
            topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry) or "Unknown"
            ckey = QABlockGenerationAgent._can(topic_name)

            summary_obj = summaries_by_can.get(ckey)
            if summary_obj is None:
                accumulated_errs.append(f"[QABlocks] No summary for '{topic_name}'; skipping topic this round.")
                covered.add(ckey)
                continue

            # Collect deep-dive nodes (preserve order)
            deep_dive_nodes: List[dict] = []
            for node in (topic_dict.get("nodes") or []):
                qtype = str(node.get("question_type", "")).strip().lower()
                if qtype in DEEP_DIVE_ALIASES:
                    deep_dive_nodes.append(node)

            if not deep_dive_nodes:
                accumulated_errs.append(f"[QABlocks] Topic '{topic_name}' has no deep-dive nodes; skipping this round.")
                covered.add(ckey)
                continue

            summary_json = json.dumps(summary_obj.model_dump() if hasattr(summary_obj, "model_dump") else summary_obj)
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
                accumulated_errs.append(f"[{topic_name}] model returned 0 QA blocks; will retry.")
            covered.add(ckey)

        # ---- finalize ----
        if final_sets:
            state.qa_blocks = QASetsSchema(qa_sets=final_sets)
            log_info("QA block generation completed")
        else:
            # No valid blocks this pass; set None to trigger regeneration.
            state.qa_blocks = None
            log_warning("QA block generation produced no valid blocks this pass")
            if not accumulated_errs:
                accumulated_errs.append("[QABlocks] No topics produced QA blocks this attempt.")

        if accumulated_errs:
            # Append all accumulated errors to state.qa_error (fallback-friendly)
            prev = getattr(state, "qa_error", "") or ""
            state.qa_error = (prev + ("\n" if prev else "") + "\n".join(accumulated_errs)).strip()

        return state

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Decide whether to retry QA generation. Retries when:
          - qa_blocks is None (no valid blocks produced yet)
          - schema validation fails
          - at least one topic has 0 blocks after validation
        """
        global count

        if getattr(state, "qa_blocks", None) is None:
            log_retry_iteration(
                reason="qa_blocks is None (no valid blocks yet); retrying",
                iteration=count,
                extra=None,
            )
            count += 1
            return True

        # Validate container schema
        try:
            QASetsSchema.model_validate(
                state.qa_blocks.model_dump() if hasattr(state.qa_blocks, "model_dump") else state.qa_blocks
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
                reason="Schema validation failed",
                iteration=count,
                extra={"error": str(ve)},
            )
            count += 1
            return True

        # Ensure every QA set has at least one block
        try:
            sets = state.qa_blocks.qa_sets if hasattr(state.qa_blocks, "qa_sets") else state.qa_blocks.get("qa_sets", [])
            if any(not (qs.get("qa_blocks") if isinstance(qs, dict) else qs.qa_blocks) for qs in sets):
                log_retry_iteration(
                    reason="At least one topic has 0 qa_blocks after validation; retrying",
                    iteration=count,
                    extra=None,
                )
                count += 1
                return True
        except Exception as e:
            log_retry_iteration(
                reason="Introspection failed while checking qa_sets; allowing retry",
                iteration=count,
                extra={"error": str(e)},
            )
            count += 1
            return True

        return False

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Graph for QA block generation:
        START -> qablock_generator -> (should_regenerate ? qablock_generator : END)
        """
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
        gb.add_edge(START, "qablock_generator")
        gb.add_conditional_edges(
            "qablock_generator",
            QABlockGenerationAgent.should_regenerate,
            {True: "qablock_generator", False: END},
        )
        return gb.compile(checkpointer=checkpointer, name="QA Block Generation Agent")


if __name__ == "__main__":
    graph = QABlockGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
