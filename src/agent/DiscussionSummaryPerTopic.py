import json
import asyncio
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from src.mongo_tools import get_mongo_tools
from string import Template
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


count = 1 


# # At top of file (if you added the log helpers there)
# def _log_planned_tool_calls(ai_msg):
#     for tc in getattr(ai_msg, "tool_calls", []) or []:
#         try:
#             print(f"[ToolCall] name={tc['name']} args={tc.get('args')}")
#         except Exception:
#             print(f"[ToolCall] {tc}")

# def _log_recent_tool_results(messages):
#     i = len(messages) - 1
#     j = False
#     while i >= 0 and getattr(messages[i], "type", None) == "tool":
#         if j == False:
#             print("----------------Nodes Tool Call logs-----------------------------------")
#             j = True
#         tm = messages[i]
#         print(f"[ToolResult] tool_call_id={getattr(tm, 'tool_call_id', None)} result={tm.content}")
#         i -= 1

# ---------- Inner ReAct state for per-topic Mongo loop ----------
class _PerTopicState(MessagesState):
    final_response: DiscussionSummaryPerTopicSchema.DiscussionTopic


class PerTopicDiscussionSummaryGenerationAgent:
    llm_dts = llm_dts

    # Flat list of Mongo tools; bind to model for tool-calling
    # Inside class TopicGenerationAgent
    # make sure you have this:
    MONGO_TOOLS = get_mongo_tools(llm=llm_dts)
    _AGENT_MODEL = llm_dts.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_dts.with_structured_output(
        DiscussionSummaryPerTopicSchema.DiscussionTopic, method="function_calling"
    )

    # ---------- Inner graph (agent -> tools -> agent ... -> respond) ----------
    @staticmethod
    def _agent_node(state: _PerTopicState):
        # _log_recent_tool_results(state["messages"])   # optional logging
        # If we just came from ToolNode, the last messages are ToolMessages → print them.
        log_tool_activity(
            messages=state["messages"],
            ai_msg=None,
            agent_name=AGENT_NAME,
            logger=LOGGER,
            header="Discussion Summary Tool Activity",
            pretty_json=True
        )

        ai = PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    def _respond_node(state: _PerTopicState):
        msgs = state["messages"]

        # Find the last ASSISTANT message that does NOT request tools
        ai_content = None
        for m in reversed(msgs):
            # LangChain AssistantMessage typically has .type == "ai"
            if getattr(m, "type", None) in ("ai", "assistant"):
                if not getattr(m, "tool_calls", None):  # completed answer
                    ai_content = m.content
                    break

        # Fallbacks: if none found, take the last assistant content even if it had tool calls,
        # else take the last message content.
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
        # last = state["messages"][-1]
        # if getattr(last, "tool_calls", None):
        #     _log_planned_tool_calls(last)  # optional logging
        #     return "continue"
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            # Log planned tool calls from the assistant message
            log_tool_activity(
                messages=state["messages"],
                ai_msg=last,
                agent_name=AGENT_NAME,
                logger=LOGGER,
                header="Discussion Summary Tool Activity",
                pretty_json=True
            )
            return "continue"
        return "respond"

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

    @staticmethod
    async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any], thread_id):
        """
        Drive inner graph for a single topic:
        System prompt -> agent (may call Mongo tools) -> respond (structured DiscussionTopic)
        """

        class AtTemplate(Template):
            delimiter = '@'

        tpl = AtTemplate(DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT)
        content = tpl.substitute(
            generated_summary=generated_summary_json,
            interview_topic=json.dumps(topic, ensure_ascii=False),
            thread_id=thread_id,
        )

        sys_message = SystemMessage(
            content=content
        )
        
        
        # This acts as the initial prompt to trigger the agent to start generating the summary 
        # and use the tools described in the SystemMessage.
        trigger_message = HumanMessage(
            content="Based on the provided instructions please start the process"
        )

        result = await PerTopicDiscussionSummaryGenerationAgent._per_topic_graph.ainvoke(
            {"messages": [sys_message, trigger_message]} # <-- Passing both System and Human messages
        )
        return result["final_response"]

    @staticmethod
    async def should_regenerate(state: AgentInternalState):
        global count 

        input_topics = {t.topic for t in state.interview_topics.interview_topics}
        output_topics = {dt.topic for dt in state.discussion_summary_per_topic.discussion_topics}
        if input_topics != output_topics:
            missing = input_topics - output_topics
            extra = output_topics - input_topics
            print(f"[PerTopic] Topic mismatch: missing {missing}, extra {extra}")
            # print(f"Topic wise Discussion Summary Retry Iteration -> {count}")
            log_retry_iteration(
                                    agent_name=AGENT_NAME,
                                    iteration=count,                          
                                    reason="Topic mismatch",
                                    logger=LOGGER,
                                    pretty_json=True,
                                    extra={"missing": missing, "extra": extra}
                                )
            count += 1
            return True
        else:
            return False

    @staticmethod
    async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Parallel per-topic summaries using asyncio.gather (inner graph handles tool-calls and structuring).
        Produces DiscussionSummaryPerTopicSchema with one entry per input topic.
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
                if hasattr(r, "model_copy"):           # pydantic v2
                    r = r.model_copy(update={"topic": in_topic})
                elif hasattr(r, "copy"):               # pydantic v1
                    r = r.copy(update={"topic": in_topic})
                elif isinstance(r, dict):
                    r["topic"] = in_topic
                else:
                    setattr(r, "topic", in_topic)
                discussion_topics.append(r)
            except Exception as e:
                print(f"Topic {idx}: could not append structured response ({e})")

        state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
            discussion_topics=discussion_topics
        )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
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
