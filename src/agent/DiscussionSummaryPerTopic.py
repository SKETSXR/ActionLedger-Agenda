import json
import asyncio
from typing import List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from string import Template

from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import DiscussionSummaryPerTopicSchema
from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
    DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT,
)
from ..model_handling import llm_dts
from ..logging_tools import get_tool_logger, log_tool_activity, log_retry_iteration


AGENT_NAME = "discussion_summary_agent"
LOG_DIR = "logs"
LOGGER = get_tool_logger(AGENT_NAME, log_dir=LOG_DIR, backup_count=365)

# Global retry counter used for logging iteration counts
count = 1


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
        log_tool_activity(
            messages=state["messages"],
            ai_msg=None,
            agent_name=AGENT_NAME,
            logger=LOGGER,
            header="Discussion Summary Tool Activity",
            pretty_json=True,
        )
        ai = PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.invoke(state["messages"])
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

        final_obj = PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.invoke(
            [HumanMessage(content=ai_content)]
        )
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _PerTopicState):
        """
        Router: if the latest assistant message contains tool calls,
        continue to the ToolNode; otherwise, proceed to respond.
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(
                messages=state["messages"],
                ai_msg=last,
                agent_name=AGENT_NAME,
                logger=LOGGER,
                header="Discussion Summary Tool Activity",
                pretty_json=True,
            )
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
                agent_name=AGENT_NAME,
                iteration=count,
                reason="Topic mismatch",
                logger=LOGGER,
                pretty_json=True,
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
                # Keep processing the rest; this item failed to update/append
                LOGGER.warning("Failed to append structured response for topic index %s: %s", idx, e)

        state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
            discussion_topics=discussion_topics
        )
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
