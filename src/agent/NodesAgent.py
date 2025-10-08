import json
import copy
import logging
import os
import sys
from typing import List, Any, Dict, Optional, Sequence
from datetime import datetime, date

from logging.handlers import TimedRotatingFileHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError
from string import Template

from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import NodesSchema, TopicWithNodesSchema
from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
from ..model_handling import llm_n


# ==============================
# Config (env-overridable)
# ==============================

AGENT_NAME = "nodes_agent"

LOG_DIR = os.getenv("NODES_AGENT_LOG_DIR", "logs")
LOG_LEVEL = getattr(
    logging,
    os.getenv("NODES_AGENT_LOG_LEVEL", "INFO").upper(),
    logging.INFO,
)
LOG_FILE = os.getenv("NODES_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("NODES_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("NODES_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("NODES_AGENT_LOG_BACKUP_COUNT", "365"))

# Global retry counter used only for logging iteration counts.
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


def _fmt_full(val: Any) -> str:
    """Return full string; pretty-print JSON strings/objects when possible."""
    try:
        if isinstance(val, (dict, list)):
            import json as _json
            return _json.dumps(val, ensure_ascii=False, indent=2)
        if isinstance(val, str):
            s = val.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                import json as _json
                return _json.dumps(_json.loads(s), ensure_ascii=False, indent=2)
        return str(val)
    except Exception:
        return str(val)


def log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    """
    Human-readable tool activity:
      - plans (from the assistant msg's tool_calls)
      - results (returns the tool messages at the end)
    """
    if not messages:
        return

    # Planned tool calls
    tool_calls = getattr(ai_msg, "tool_calls", None)
    if tool_calls:
        log_info("Tool plan:")
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
            log_info(f"  planned -> {name} args={_fmt_full(args)}")

    # Recent tool results (messages end with tool outputs)
    i = len(messages) - 1
    printed = False
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        if not printed:
            log_info("Tool results:")
            printed = True
        tm = messages[i]
        log_info(
            f"  result <- id={getattr(tm, 'tool_call_id', None)} "
            f"data={_fmt_full(getattr(tm, 'content', None))}"
        )
        i -= 1


def log_retry_iteration(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    """Human-format retry line."""
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


# ---------- Inner ReAct state for Mongo loop (per-topic) ----------
class _MongoNodesState(MessagesState):
    """State container for the inner ReAct loop that generates nodes per topic."""
    final_response: TopicWithNodesSchema


class NodesGenerationAgent:
    """
    Generates per-topic node structures by running a tool-using inner ReAct loop
    (with Mongo tools), coercing the final assistant content into
    TopicWithNodesSchema, and validating the overall NodesSchema container.
    """

    # LLM & tools (kept as class attributes to reuse clients and bindings)
    llm_n = llm_n
    MONGO_TOOLS = get_mongo_tools(llm=llm_n)

    _AGENT_MODEL = llm_n.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_n.with_structured_output(
        TopicWithNodesSchema, method="function_calling"
    )

    # ---------- Inner graph (agent -> tools -> agent ... -> respond) ----------
    @staticmethod
    def _agent_node(state: _MongoNodesState):
        """
        Invoke the tool-enabled model. If the assistant plans tool calls,
        LangGraph will execute them via the ToolNode.
        """
        log_info("Calling LLM (agent)")
        log_tool_activity(state["messages"], ai_msg=None)
        ai = NodesGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        log_info("LLM (agent) call succeeded")
        return {"messages": [ai]}

    @staticmethod
    def _respond_node(state: _MongoNodesState):
        """
        Take the most recent assistant message without tool calls and coerce it
        into TopicWithNodesSchema using the structured-output model. If none exists,
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
        final_obj = NodesGenerationAgent._STRUCTURED_MODEL.invoke([HumanMessage(content=ai_content)])
        log_info("LLM (structured) call succeeded")
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoNodesState):
        """
        Router node: if the latest assistant message contains tool calls,
        continue to the ToolNode, otherwise, proceed to respond.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # Compile inner ReAct graph once
    _workflow = StateGraph(_MongoNodesState)
    _workflow.add_node("agent", _agent_node)
    _workflow.add_node("respond", _respond_node)
    _workflow.add_node("tools", ToolNode(MONGO_TOOLS, tags=["mongo-tools+arith-tools"]))
    _workflow.set_entry_point("agent")
    _workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"continue": "tools", "respond": "respond"},
    )
    _workflow.add_edge("tools", "agent")
    _workflow.add_edge("respond", END)
    _nodes_graph = _workflow.compile()

    # ----------------- utilities -----------------
    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        """Best-effort conversion to a dict."""
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "dict"):
            return x.dict()
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _get_topic_name(obj: Any) -> str:
        """Extract a human-friendly topic name from common keys; default to 'Unknown'."""
        d = NodesGenerationAgent._as_dict(obj)
        for k in ("topic", "name", "title", "label"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    # ----------------- utilities -----------------
    @staticmethod
    def _to_primitive(x: Any) -> Any:
        """Recursively convert Pydantic models and other objects to JSON-serializable primitives."""
        # Pydantic v2
        if hasattr(x, "model_dump"):
            try:
                return NodesGenerationAgent._to_primitive(x.model_dump())
            except Exception:
                pass
        # Pydantic v1
        if hasattr(x, "dict"):
            try:
                return NodesGenerationAgent._to_primitive(x.dict())
            except Exception:
                pass

        if isinstance(x, dict):
            return {k: NodesGenerationAgent._to_primitive(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [NodesGenerationAgent._to_primitive(v) for v in x]
        if isinstance(x, (datetime, date)):
            return x.isoformat()
        # Basic JSON types pass through
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        # Fallback: try __dict__ or string
        if hasattr(x, "__dict__"):
            return NodesGenerationAgent._to_primitive(vars(x))
        return str(x)

    @staticmethod
    def _to_json_one(x: Any) -> str:
        """Stable JSON for an arbitrary object (handles nested Pydantic models)."""
        return json.dumps(NodesGenerationAgent._to_primitive(copy.deepcopy(x)), ensure_ascii=False)


    # ----------------- utilities -----------------

    @staticmethod
    async def _gen_once(
        per_topic_summary_json: str,
        thread_id,
        nodes_error,
    ) -> TopicWithNodesSchema:
        """
        Generate one TopicWithNodesSchema by running the inner ReAct loop with a
        system prompt constructed from the per-topic summary.
        """
        class AtTemplate(Template):
            delimiter = "@"

        tpl = AtTemplate(NODES_AGENT_PROMPT)
        content = tpl.substitute(
            per_topic_summary_json=per_topic_summary_json,
            thread_id=thread_id,
            nodes_error=nodes_error,
        )

        sys_message = SystemMessage(content=content)
        trigger_message = HumanMessage(content="Based on the provided instructions please start the process")

        # Runs inner graph: agent <-> tools ... -> respond (structured)
        result = await NodesGenerationAgent._nodes_graph.ainvoke(
            {"messages": [sys_message, trigger_message]}
        )
        return result["final_response"]

    # ----------------------------- Main node generation -----------------------------
    @staticmethod
    async def nodes_generator(state: AgentInternalState) -> AgentInternalState:
        """
        For each (topic, summary) pair:
          - validate total_questions per topic
          - run the inner ReAct loop to produce TopicWithNodesSchema
        Aggregates results into NodesSchema on the state.
        """
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")

        topics_list = list(state.interview_topics.interview_topics)
        try:
            summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
        except Exception:
            summaries_list = list(state.discussion_summary_per_topic)

        pair_count = min(len(topics_list), len(summaries_list))
        if pair_count == 0:
            raise ValueError("No topics/summaries to process.")

        # Snapshot upstream summaries to ensure no mutation occurs
        snapshot = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True,
        )

        out: List[TopicWithNodesSchema] = []

        for dspt_obj in zip(topics_list, summaries_list):
            per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)

            resp = await NodesGenerationAgent._gen_once(
                per_topic_summary_json, state.id, state.nodes_error
            )
            out.append(resp)

        # Verify upstream summary wasn’t mutated
        after = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True,
        )
        if after != snapshot:
            raise RuntimeError("discussion_summary_per_topic mutated during node generation")

        state.nodes = NodesSchema(
            topics_with_nodes=[t.model_dump() if hasattr(t, "model_dump") else t for t in out]
        )
        log_info("Nodes generation completed")
        return state

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Decide whether to regenerate node structures. Retries when:
          - no nodes were produced yet
          - container schema (NodesSchema) is invalid
          - any TopicWithNodesSchema item inside is invalid
        """
        global count

        if getattr(state, "nodes", None) is None:
            return True

        # Validate NodesSchema (container)
        try:
            NodesSchema.model_validate(
                state.nodes.model_dump() if hasattr(state.nodes, "model_dump") else state.nodes
            )
        except ValidationError as ve:
            state.nodes_error += (
                "The previous generated o/p did not follow the given schema as it got following errors:\n"
                + (getattr(state, "nodes_error", "") or "")
                + "\n[NodesSchema ValidationError]\n"
                + str(ve)
                + "\n"
            )
            log_retry_iteration(
                reason=f"[NodesGen][ValidationError] Container NodesSchema invalid\n {ve}",
                iteration=count,
            )
            count += 1
            return True

        # Validate each topic payload
        try:
            topics_payload = (
                state.nodes.topics_with_nodes
                if hasattr(state.nodes, "topics_with_nodes")
                else state.nodes.get("topics_with_nodes", [])
            )
        except Exception as e:
            state.nodes_error += "\n[NodesSchema Payload Error]\n" + str(e) + "\n"
            log_retry_iteration(
                reason=f"[NodesGen][ValidationError] Could not read topics_with_nodes: {e}",
                iteration=count,
            )
            count += 1
            return True

        any_invalid = False
        for idx, item in enumerate(topics_payload):
            try:
                TopicWithNodesSchema.model_validate(
                    item.model_dump() if hasattr(item, "model_dump") else item
                )
            except ValidationError as ve:
                any_invalid = True
                state.nodes_error += f"\n[TopicWithNodesSchema ValidationError idx={idx}]\n{ve}\n"

        if any_invalid:
            log_retry_iteration(
                reason="node schema error",
                iteration=count,
            )
            count += 1
        return any_invalid

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Graph for node generation:
        START -> nodes_generator -> (should_regenerate ? nodes_generator : END)
        """
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
        gb.add_edge(START, "nodes_generator")
        gb.add_conditional_edges(
            "nodes_generator",
            NodesGenerationAgent.should_regenerate,
            {True: "nodes_generator", False: END},
        )
        return gb.compile(checkpointer=checkpointer, name="Nodes Generation Agent")


if __name__ == "__main__":
    graph = NodesGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
