# Running with schema check and regenerate
import json
import copy
from typing import List, Any, Dict

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
from ..logging_tools import get_tool_logger, log_tool_activity, log_retry_iteration


# Global retry counter used only for logging iteration counts.
count = 1

AGENT_NAME = "nodes_agent"
LOG_DIR = "logs"
LOGGER = get_tool_logger(AGENT_NAME, log_dir=LOG_DIR, backup_count=365)


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
        log_tool_activity(
            state["messages"],
            ai_msg=None,
            agent_name=AGENT_NAME,
            logger=LOGGER,
            header="Nodes Tool Activity",
            pretty_json=True,
        )
        ai = NodesGenerationAgent._AGENT_MODEL.invoke(state["messages"])
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

        final_obj = NodesGenerationAgent._STRUCTURED_MODEL.invoke([HumanMessage(content=ai_content)])
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoNodesState):
        """
        Router node: if the latest assistant message contains tool calls,
        continue to the ToolNode; otherwise, proceed to respond.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(
                state["messages"],
                ai_msg=last,
                agent_name=AGENT_NAME,
                logger=LOGGER,
                header="Nodes Tool Activity",
                pretty_json=True,
            )
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
        """Best-effort conversion to a dict without altering semantics."""
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

    @staticmethod
    def _to_json_one(x: Any) -> str:
        """Deep-copy an object into a stable JSON string without mutating upstream structures."""
        if hasattr(x, "model_dump"):
            return json.dumps(copy.deepcopy(x.model_dump()))
        if hasattr(x, "dict"):
            return json.dumps(copy.deepcopy(x.dict()))
        return json.dumps(copy.deepcopy(x))

    @staticmethod
    def _get_total_questions(topic_obj: Any, _dspt_obj: Any) -> int:
        """
        Return the per-topic total_questions from the interview_topics entry ONLY.
        Enforces >= 4 because at least one Deep Dive (threshold >= 2) is required:
          total_questions = 1 (Opening) + 1 (Direct) + sum(Deep Dive thresholds >= 2) → minimum 4.
        """
        d = NodesGenerationAgent._as_dict(topic_obj)
        tq = d.get("total_questions")
        if not isinstance(tq, int) or tq < 4:
            raise ValueError(
                "Each topic must set total_questions >= 4 because at least 1 Deep Dive (min threshold 2) is required."
            )
        return tq

    @staticmethod
    async def _gen_once(
        per_topic_summary_json: str,
        total_no_questions_topic,  # kept for interface compatibility
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

        for topic_obj, dspt_obj in zip(topics_list, summaries_list):
            total_no_questions_topic = NodesGenerationAgent._get_total_questions(topic_obj, dspt_obj)
            per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)

            resp = await NodesGenerationAgent._gen_once(
                per_topic_summary_json, total_no_questions_topic, state.id, state.nodes_error
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
                agent_name=AGENT_NAME,
                iteration=count,
                reason=f"[NodesGen][ValidationError] Container NodesSchema invalid\n {ve}",
                logger=LOGGER,
                pretty_json=True,
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
                agent_name=AGENT_NAME,
                iteration=count,
                reason=f"[NodesGen][ValidationError] Could not read topics_with_nodes: {e}",
                logger=LOGGER,
                pretty_json=True,
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
                agent_name=AGENT_NAME,
                iteration=count,
                reason="node schema error",
                logger=LOGGER,
                pretty_json=True,
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
