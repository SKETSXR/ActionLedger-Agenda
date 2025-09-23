import json
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.tools import tool
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import CollectiveInterviewTopicSchema, CollectiveInterviewTopicFeedbackSchema
from ..schema.input_schema import SkillTreeSchema
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
# from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_SELF_REFLECTION_PROMPT
from ..model_handling import llm_tg
from src.mongo_tools import get_mongo_tools

# set_llm_cache(InMemoryCache())


class TopicGenerationAgent:

    llm_tg = llm_tg
    MONGO_TOOLS = get_mongo_tools(llm=llm_tg)
    llm_tg_with_tools = llm_tg.bind_tools(MONGO_TOOLS)
    
    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        interview_topics_feedback = state.interview_topics_feedback.feedback if state.interview_topics_feedback is not None else ""
        state.interview_topics_feedbacks += "\n" + interview_topics_feedback + "\n"

        response = await TopicGenerationAgent.llm_tg_with_tools \
        .with_structured_output(CollectiveInterviewTopicSchema, method="function_calling") \
        .ainvoke(
            [
                SystemMessage(
                    content=TOPIC_GENERATION_AGENT_PROMPT.format(
                        generated_summary=state.generated_summary.model_dump_json(),
                        interview_topics_feedbacks=state.interview_topics_feedbacks,
                        thread_id=state.id
                    )
                ),
                state.messages[-1] if len(state.messages) else ""
            ]
        )
        state.interview_topics = response
        return state

    
    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:

        def level3_leaves(root: SkillTreeSchema) -> list[SkillTreeSchema]:
            if not root.children:
                return []
            leaves: list[SkillTreeSchema] = []
            skills_priority_must: list[SkillTreeSchema] = []
            for domain in root.children:                 
                for leaf in (domain.children or []):
                    if not leaf.children:  # only pick true leaves
                        if leaf.priority == "must":
                            skills_priority_must.append(leaf)
                        leaves.append(leaf)
            # return leaves
            return skills_priority_must

        # all_skill_leaves = [leaf.name for leaf in level3_leaves(state.skill_tree)]
        skills_priority_must = [leaf.name for leaf in level3_leaves(state.skill_tree)]
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
            print("Total questions in topic list does not match as decided by summary... regenerating topics...")
            return False
        # focus_area_list = all_focus_skills(state.interview_topics)

        # print(f"Skill Tree List {all_skill_leaves}")

        # print(f"\nFocus Area List {focus_area_list}")
        # for i in focus_area_list:
        #     if i not in all_skill_leaves:
        #         return False

        # response = await TopicGenerationAgent.llm_tg_with_tools \
        # .with_structured_output(CollectiveInterviewTopicFeedbackSchema, method="function_calling") \
        # .ainvoke(
        #     [
        #         SystemMessage(
        #             content=TOPIC_GENERATION_SELF_REFLECTION_PROMPT.format(
        #                 generated_summary=state.generated_summary.model_dump_json(),
        #                 interview_topics=state.interview_topics.model_dump_json(),
        #                 thread_id=state.id
        #             )
        #         )
        #     ]
        # )
        # state.interview_topics_feedback = ""
        skill_list = ""
        for i in set(skills_priority_must):
            if i not in set(focus_area_list):
                skill_list += ", " + i

        feedback = ""
        if skill_list != "":
            if state.interview_topics_feedback is not None:
                feedback = state.interview_topics_feedback.feedback
            feedback += f"Add the list of missing `must` priority skills: {skill_list} to the topic which is related to the General Skill Assessment"
            state.interview_topics_feedback = {"satisfied": False, "feedback": feedback}
            # state.interview_topics_feedback.feedback += f"Add the list of missing `must` priority skills: {skill_list} to the topic which is related to the General Skill Assessment"
            # state.interview_topics_feedback.satisfied = False
            return False
        # state.interview_topics_feedback = response
        # print(response)
        # if not state.interview_topics_feedback.satisfied:
        #     return False
        # else:
        #     return True
        return True

    @staticmethod
    def get_graph(checkpointer=None):

        graph_builder = StateGraph(
            state_schema=AgentInternalState
        )

        graph_builder.add_node("topic_generator", TopicGenerationAgent.topic_generator)
        graph_builder.add_edge(START, "topic_generator")
        graph_builder.add_conditional_edges(
            "topic_generator",
            TopicGenerationAgent.should_regenerate,
            {
                True: END,
                False: "topic_generator"
            }
        )
       
        graph = graph_builder.compile(checkpointer=checkpointer, name="Topic Generation Agent")

        return graph


if __name__ == "__main__":
    graph = TopicGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
