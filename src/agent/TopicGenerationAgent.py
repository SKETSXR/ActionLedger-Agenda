# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import SystemMessage
# from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache
# from langchain_core.tools import tool
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import CollectiveInterviewTopicSchema
# from ..schema.input_schema import SkillTreeSchema
# from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
# # from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_SELF_REFLECTION_PROMPT
# from ..model_handling import llm_tg
# from src.mongo_tools import get_mongo_tools

# # set_llm_cache(InMemoryCache())


# class TopicGenerationAgent:

#     llm_tg = llm_tg
#     MONGO_TOOLS = get_mongo_tools(llm=llm_tg)
#     llm_tg_with_tools = llm_tg.bind_tools(MONGO_TOOLS)
    
#     @staticmethod
#     async def topic_generator(state: AgentInternalState) -> AgentInternalState:
#         if not state.generated_summary:
#             raise ValueError("Summary cannot be null.")

#         interview_topics_feedback = state.interview_topics_feedback.feedback if state.interview_topics_feedback is not None else ""
#         state.interview_topics_feedbacks += "\n" + interview_topics_feedback + "\n"

#         response = await TopicGenerationAgent.llm_tg_with_tools \
#         .with_structured_output(CollectiveInterviewTopicSchema, method="function_calling") \
#         .ainvoke(
#             [
#                 SystemMessage(
#                     content=TOPIC_GENERATION_AGENT_PROMPT.format(
#                         generated_summary=state.generated_summary.model_dump_json(),
#                         interview_topics_feedbacks=state.interview_topics_feedbacks,
#                         thread_id=state.id
#                     )
#                 )
#                 # state.messages[-1] if len(state.messages) else ""
#             ]
#         )
#         state.interview_topics = response
#         return state

    
#     @staticmethod
#     async def should_regenerate(state: AgentInternalState) -> bool:

#         def level3_leavesp(root: SkillTreeSchema) -> list[SkillTreeSchema]:
#             if not root.children:
#                 return []
#             skills_priority_must: list[SkillTreeSchema] = []
#             for domain in root.children:                 
#                 for leaf in (domain.children or []):
#                     if not leaf.children:  # only pick true leaves
#                         if leaf.priority == "must":
#                             skills_priority_must.append(leaf)

#             return skills_priority_must

#         def level3_leaves(root: SkillTreeSchema) -> list[SkillTreeSchema]:
#             if not root.children:
#                 return []
#             leaves: list[SkillTreeSchema] = []
#             for domain in root.children:                 
#                 for leaf in (domain.children or []):
#                     if not leaf.children:  # only pick true leaves
#                         leaves.append(leaf)
#             return leaves


#         all_skill_leaves = [leaf.name for leaf in level3_leaves(state.skill_tree)]
#         skills_priority_must = [leaf.name for leaf in level3_leavesp(state.skill_tree)]
#         # print(skills_priority_must)
#         # print(all_skill_leaves)

#         # print(state.interview_topics.model_dump())
#         focus_area_list = []
#         for t in state.interview_topics.interview_topics:
#             for i in t.focus_area:
#                 # print(i.model_dump())
#                 for j in i.model_dump():
#                     if j == "skill":
#                         x = i.model_dump()
#                         if x["skill"] not in focus_area_list:
#                             focus_area_list.append(x["skill"]) 

#         total_questions_sum = sum(t.total_questions for t in state.interview_topics.interview_topics)
#         # print(f"Total Questions Sum: {total_questions_sum}\nTotal Questions in Summary: {state.generated_summary.total_questions}")
#         # print(focus_area_list)
#         if total_questions_sum != state.generated_summary.total_questions:
#             print("Total questions in topic list does not match as decided by summary... regenerating topics...")
#             return False
#         # focus_area_list = all_focus_skills(state.interview_topics)

#         # print(f"Skill Tree List {all_skill_leaves}")

#         # print(f"\nFocus Area List {focus_area_list}")
#         for i in focus_area_list:
#             if i not in all_skill_leaves:
#                 return False

#         # response = await TopicGenerationAgent.llm_tg_with_tools \
#         # .with_structured_output(CollectiveInterviewTopicFeedbackSchema, method="function_calling") \
#         # .ainvoke(
#         #     [
#         #         SystemMessage(
#         #             content=TOPIC_GENERATION_SELF_REFLECTION_PROMPT.format(
#         #                 generated_summary=state.generated_summary.model_dump_json(),
#         #                 interview_topics=state.interview_topics.model_dump_json(),
#         #                 thread_id=state.id
#         #             )
#         #         )
#         #     ]
#         # )
#         # state.interview_topics_feedback = ""
#         skill_list = ""
#         for i in set(skills_priority_must):
#             if i not in set(focus_area_list):
#                 skill_list += ", " + i

#         feedback = ""
#         if skill_list != "":
#             if state.interview_topics_feedback is not None:
#                 feedback = state.interview_topics_feedback.feedback
#             feedback += f"Please keep the topic set as it is irresepective of below instructions: ```\n{state.interview_topics.model_dump()}```\n But add the list of missing `must` priority skills: \n{skill_list}\n to the focus areas of the last topic which being General Skill Assessment"
#             state.interview_topics_feedback = {"satisfied": False, "feedback": feedback}
#             # state.interview_topics_feedback.feedback += f"Add the list of missing `must` priority skills: {skill_list} to the topic which is related to the General Skill Assessment"
#             # state.interview_topics_feedback.satisfied = False
#             return False
#         # state.interview_topics_feedback = response
#         # print(response)
#         # if not state.interview_topics_feedback.satisfied:
#         #     return False
#         # else:
#         #     return True
#         return True

#     @staticmethod
#     def get_graph(checkpointer=None):

#         graph_builder = StateGraph(
#             state_schema=AgentInternalState
#         )

#         graph_builder.add_node("topic_generator", TopicGenerationAgent.topic_generator)
#         graph_builder.add_edge(START, "topic_generator")
#         graph_builder.add_conditional_edges(
#             "topic_generator",
#             TopicGenerationAgent.should_regenerate,
#             {
#                 True: END,
#                 False: "topic_generator"
#             }
#         )
       
#         graph = graph_builder.compile(checkpointer=checkpointer, name="Topic Generation Agent")

#         return graph


# if __name__ == "__main__":
#     graph = TopicGenerationAgent.get_graph()
#     g = graph.get_graph().draw_ascii()
#     print(g)


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langgraph.graph import MessagesState
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import CollectiveInterviewTopicSchema
from ..schema.input_schema import SkillTreeSchema
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from ..model_handling import llm_tg
from src.mongo_tools import get_mongo_tools

# set_llm_cache(InMemoryCache())

i = 1
# At top of file (if you added the log helpers there)
def _log_planned_tool_calls(ai_msg):
    for tc in getattr(ai_msg, "tool_calls", []) or []:
        try:
            print(f"[ToolCall] name={tc['name']} args={tc.get('args')}")
        except Exception:
            print(f"[ToolCall] {tc}")

def _log_recent_tool_results(messages):
    i = len(messages) - 1
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        tm = messages[i]
        print(f"[ToolResult] tool_call_id={getattr(tm, 'tool_call_id', None)} result={tm.content}")
        i -= 1



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
        print("--------------Topic Tool Call logs------------------")
        _log_recent_tool_results(state["messages"])   # optional logging

        ai = TopicGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        return {"messages": [ai]}

    # @staticmethod
    # def _respond_node(state: _MongoAgentState):
    #     """Guarantee final JSON using structured model."""
    #     # Prefer the latest ToolMessage content (Mongo result); fallback to last AI content
    #     msgs = state["messages"]
    #     content = None
    #     for m in reversed(msgs):
    #         if getattr(m, "type", None) == "tool" or m.__class__.__name__ == "ToolMessage":
    #             content = m.content
    #             break
    #     if content is None:
    #         content = msgs[-1].content
    #     final_obj = TopicGenerationAgent._STRUCTURED_MODEL.invoke([HumanMessage(content=content)])
    #     return {"final_response": final_obj}

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
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            _log_planned_tool_calls(last)  # optional logging
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

    # ---------- Your original node, now using the inner graph ----------
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
        # MONGO_USAGE_RULES = '''
        #                     MongoDB tool usage rules (STRICT):
        #                     - Always use the right tool:
        #                     • mongodb_list_collections()          # no args
        #                     • mongodb_schema(collection_names)    # comma-separated names: "cv, summary"
        #                     • mongodb_query(collection, query)    # BOTH args MUST be given
        #                     - 'collection' is the collection name (e.g., "cv", "summary", "question_guidelines").
        #                     - 'query' is a pure JSON filter string ONLY (no 'db.' shell, no aggregate, no comments).
        #                     Example: collection="summary", query='{"_id": "thread_38"}'
        #                     - Do NOT send just a JSON object as the whole command.
        #                     - Do NOT send 'db.collection.find(...)' - that will fail. Use the two-arg format above.
        #                     - If a query fails twice, STOP calling tools and proceed with the data you have.\n
        #                     '''
        MONGO_USAGE_RULES = ""
        from string import Template

        class AtTemplate(Template):
            delimiter = '@'   # anything not used in your prompt samples

        tpl = AtTemplate(TOPIC_GENERATION_AGENT_PROMPT)
        content = tpl.substitute(
            generated_summary=state.generated_summary.model_dump_json(),
            interview_topics_feedbacks=state.interview_topics_feedbacks,
            thread_id=state.id,
        )


        messages = [
            SystemMessage(
                content=content
            )
        ]

        # Let LangGraph handle tool-calling loop (agent <-> tools), then structure the output
        result = await TopicGenerationAgent._mongo_graph.ainvoke({"messages": messages})
        state.interview_topics = result["final_response"]
        return state


    # @staticmethod
    # async def should_regenerate(state: AgentInternalState) -> bool:
    #     global i 
    #     print(f"Topic Iteration -> {i}")
    #     i += 1
    #     def level3_leavesp(root: SkillTreeSchema) -> list[SkillTreeSchema]:
    #         if not root.children: return []
    #         out = []
    #         for domain in root.children:
    #             for leaf in (domain.children or []):
    #                 if not leaf.children and leaf.priority == "must":
    #                     out.append(leaf)
    #         return out

    #     def level3_leaves(root: SkillTreeSchema) -> list[SkillTreeSchema]:
    #         if not root.children: return []
    #         out = []
    #         for domain in root.children:
    #             for leaf in (domain.children or []):
    #                 if not leaf.children:
    #                     out.append(leaf)
    #         return out

    #     all_skill_leaves = [leaf.name for leaf in level3_leaves(state.skill_tree)]
    #     must_leaves = [leaf.name for leaf in level3_leavesp(state.skill_tree)]

    #     # Gather all skills used
    #     used_skills = []
    #     for t in state.interview_topics.interview_topics:
    #         for fa in t.focus_area:
    #             s = fa.model_dump().get("skill")
    #             if s and s not in used_skills:
    #                 used_skills.append(s)

    #     # Check total questions
    #     target_total = state.generated_summary.total_questions
    #     got_total = sum(t.total_questions for t in state.interview_topics.interview_topics)
    #     if got_total != target_total:
    #         state.interview_topics_feedback = type("FB", (), {})()
    #         state.interview_topics_feedback.feedback = (
    #             f"Fix only the `total_questions` so that the sum equals {target_total}. "
    #             f"Do not change topics or focus_area; keep everything else identical."
    #         )
    #         return False

    #     # Check skill validity
    #     for s in used_skills:
    #         if s not in all_skill_leaves:
    #             state.interview_topics_feedback = type("FB", (), {})()
    #             state.interview_topics_feedback.feedback = (
    #                 f"The skill `{s}` is not a level-3 leaf. Replace it with a verbatim leaf from T."
    #             )
    #             return False

    #     # Check must coverage
    #     missing = sorted(set(must_leaves) - set(used_skills))
    #     if missing:
    #         state.interview_topics_feedback = type("FB", (), {})()
    #         state.interview_topics_feedback.feedback = (
    #             "Add the following missing `must` skills ONLY to the last topic "
    #             "(General skill assessment) without changing anything else:\n"
    #             + ", ".join(missing)
    #         )
    #         return False

    #     return True

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:

        global i 

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
            print(f"Total questions in topic list does not match as decided by summary... regenerating topics... retry iteration -> {i}")
            i += 1
            return False
        # focus_area_list = all_focus_skills(state.interview_topics)

        # print(f"Skill Tree List {all_skill_leaves}")

        # print(f"\nFocus Area List {focus_area_list}")
        for i in focus_area_list:
            if i not in all_skill_leaves:
                print(f"Topic Retry Iteration -> {i}")
                i += 1
                return False

        skill_list = ""
        for i in set(skills_priority_must):
            if i not in set(focus_area_list):
                skill_list += ", " + i

        feedback = ""
        if skill_list != "":
            if state.interview_topics_feedback is not None:
                feedback = state.interview_topics_feedback.feedback
            feedback += f"Please keep the topic set as it is irresepective of below instructions: ```\n{state.interview_topics.model_dump()}```\n But add the list of missing `must` priority skills: \n{skill_list}\n to the focus areas of the last topic which being General Skill Assessment"
            state.interview_topics_feedback = {"satisfied": False, "feedback": feedback}
            print(f"Topic Retry Iteration -> {i}")
            i += 1

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
