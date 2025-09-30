
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState
from string import Template
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import CollectiveInterviewTopicSchema
from ..schema.input_schema import SkillTreeSchema
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from ..model_handling import llm_tg
from src.mongo_tools import get_mongo_tools
from ..logging_tools import get_tool_logger, log_tool_activity, log_retry_iteration

# set_llm_cache(InMemoryCache())

count = 1

# ---------------- Logger config ----------------
AGENT_NAME = "topic_generation_agent"
LOG_DIR = "logs"
LOGGER = get_tool_logger(AGENT_NAME, log_dir=LOG_DIR, backup_count=365)

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


# ---------- Inner ReAct state for Mongo loop ----------
class _MongoAgentState(MessagesState):
    final_response: CollectiveInterviewTopicSchema


class TopicGenerationAgent:

    # Models & tools
    llm_tg = llm_tg
    # Inside class TopicGenerationAgent
    MONGO_TOOLS = get_mongo_tools(llm=llm_tg)
    _AGENT_MODEL = llm_tg.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_tg.with_structured_output(
        CollectiveInterviewTopicSchema, method="function_calling"
    )

    # ---------- Build the inner ReAct graph ONCE ----------
    # Nodes
    @staticmethod
    def _agent_node(state: _MongoAgentState):
        """LLM picks tool(s) or answers; LangGraph will handle tool exec via ToolNode."""
        # _log_recent_tool_results(state["messages"])   # optional logging
        log_tool_activity(
            messages=state["messages"],
            ai_msg=None,
            agent_name=AGENT_NAME,
            logger=LOGGER,
            header="Topic Generation Tool Activity",
            pretty_json=True 
        )

        ai = TopicGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    def _respond_node(state: _MongoAgentState):
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

        final_obj = TopicGenerationAgent._STRUCTURED_MODEL.invoke(
            [HumanMessage(content=ai_content)]
        )
        return {"final_response": final_obj}


    @staticmethod
    def _should_continue(state: _MongoAgentState):
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
                header="Topic Generation Tool Activity",
                pretty_json=True 
            )
            return "continue"
        return "respond"

    # Compile inner graph
    _workflow = StateGraph(_MongoAgentState)
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
    _mongo_graph = _workflow.compile()

    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        interview_topics_feedback = (
            state.interview_topics_feedback.feedback
            if state.interview_topics_feedback is not None
            else ""
        )
        state.interview_topics_feedbacks += "\n" + interview_topics_feedback + "\n"

        class AtTemplate(Template):
            delimiter = '@'

        tpl = AtTemplate(TOPIC_GENERATION_AGENT_PROMPT)
        content = tpl.substitute(
            generated_summary=state.generated_summary.model_dump_json(),
            interview_topics_feedbacks=state.interview_topics_feedbacks,
            thread_id=state.id,
        )

        messages = [
            SystemMessage(
                content=content
            ),
            HumanMessage(
                content="Based on the instructions, please start the process."
            )
        ]

        # Let LangGraph handle tool-calling loop (agent <-> tools), then structure the output
        result = await TopicGenerationAgent._mongo_graph.ainvoke({"messages": messages})
        state.interview_topics = result["final_response"]
        return state

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:

        global count

        def level3_leavesp(root: SkillTreeSchema) -> list[SkillTreeSchema]:
            if not root.children:
                return []
            skills_priority_must: list[SkillTreeSchema] = []
            for domain in root.children:                 
                for leaf in (domain.children or []):
                    if not leaf.children:  # only pick true leaves
                        if leaf.priority == "must":
                            skills_priority_must.append(leaf)

            return skills_priority_must

        def level3_leaves(root: SkillTreeSchema) -> list[SkillTreeSchema]:
            if not root.children:
                return []
            leaves: list[SkillTreeSchema] = []
            for domain in root.children:                 
                for leaf in (domain.children or []):
                    if not leaf.children:  # only pick true leaves
                        leaves.append(leaf)
            return leaves


        all_skill_leaves = [leaf.name for leaf in level3_leaves(state.skill_tree)]
        skills_priority_must = [leaf.name for leaf in level3_leavesp(state.skill_tree)]
        # print(skills_priority_must)
        # print(all_skill_leaves)

        # print(state.interview_topics.model_dump())
        focus_area_list = []
        for t in state.interview_topics.interview_topics:
            for i in t.focus_area:
                # print(i.model_dump())
                for j in i.model_dump():
                    if j == "skill":
                        x = i.model_dump()
                        if x["skill"] not in focus_area_list:
                            focus_area_list.append(x["skill"]) 

        total_questions_sum = sum(t.total_questions for t in state.interview_topics.interview_topics)
        # print(f"Total Questions Sum: {total_questions_sum}\nTotal Questions in Summary: {state.generated_summary.total_questions}")
        # print(focus_area_list)
        if total_questions_sum != state.generated_summary.total_questions:
            # print(f"Total questions in topic list does not match as decided by summary... regenerating topics... retry iteration -> {count}")
            log_retry_iteration(
                                    agent_name=AGENT_NAME,
                                    iteration=count,
                                    reason="Total questions mismatch",
                                    logger=LOGGER,
                                    pretty_json=True,
                                    extra={
                                        "got_total": total_questions_sum,
                                        "target_total": state.generated_summary.total_questions
                                    }
                                )
            count += 1
            return False
        # focus_area_list = all_focus_skills(state.interview_topics)

        # print(f"Skill Tree List {all_skill_leaves}")

        # print(f"\nFocus Area List {focus_area_list}")
        for i in focus_area_list:
            if i not in all_skill_leaves:
                # print(f"Topic Retry Iteration -> {count}")
                log_retry_iteration(
                                        agent_name=AGENT_NAME,
                                        iteration=count,
                                        reason="Invalid focus skill",
                                        logger=LOGGER,
                                        pretty_json=True,
                                        extra={"skill": i}
                                    )
                count += 1
                return False

        skill_list = ""
        for i in set(skills_priority_must):
            if i not in set(focus_area_list):
                skill_list += ", " + i

        feedback = ""
        if skill_list != "":
            if state.interview_topics_feedback is not None:
                feedback = state.interview_topics_feedback.feedback
            feedback += f"Please keep the topic set as it is irresepective of below instructions: ```\n{state.interview_topics.model_dump()}```\n But add the list of missing `must` priority skills in this as per the \n{skill_list}\n to the focus areas of the last topic which being General Skill Assessment"
            state.interview_topics_feedback = {"satisfied": False, "feedback": feedback}
            # print(f"Topic Retry Iteration -> {count}")
            log_retry_iteration(
                                    agent_name=AGENT_NAME,
                                    iteration=count,
                                    reason="Missing MUST skills",
                                    logger=LOGGER,
                                    pretty_json=True,
                                    extra={"missing_must": skill_list}
                                )
            count += 1
            # print("-----------Topic retry logging-------------")

            return False

        return True

    @staticmethod
    def get_graph(checkpointer=None):
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node("topic_generator", TopicGenerationAgent.topic_generator)
        graph_builder.add_edge(START, "topic_generator")
        graph_builder.add_conditional_edges(
            "topic_generator",
            TopicGenerationAgent.should_regenerate,
            {True: END, False: "topic_generator"},
        )
        return graph_builder.compile(checkpointer=checkpointer, name="Topic Generation Agent")


if __name__ == "__main__":
    graph = TopicGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
