from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import CollectiveInterviewTopicSchema
from ..schema.input_schema import SkillTreeSchema
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from ..model_handling import llm_tg

set_llm_cache(InMemoryCache())


class TopicGenerationAgent:

    llm_tg = llm_tg

    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")
     
        response = await TopicGenerationAgent.llm_tg \
        .with_structured_output(CollectiveInterviewTopicSchema, method="function_calling") \
        .ainvoke(
            [
                SystemMessage(
                    content=TOPIC_GENERATION_AGENT_PROMPT.format(
                        generated_summary=state.generated_summary.model_dump_json(),
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
            for domain in root.children:                 
                for leaf in (domain.children or []):
                    if not leaf.children:  # only pick true leaves
                        leaves.append(leaf)
            return leaves

        all_skill_leaves = [leaf.name for leaf in level3_leaves(state.skill_tree)]

        focus_area_list = []
        for t in state.interview_topics.interview_topics:
            for i in list(t.focus_area.keys()):
                if i not in focus_area_list:
                    focus_area_list.append(i) 

        total_questions_sum = sum(t.total_questions for t in state.interview_topics.interview_topics)
        # print(f"Total Questions Sum: {total_questions_sum}\nTotal Questions in Summary: {state.generated_summary.total_questions}")

        if total_questions_sum != state.generated_summary.total_questions:
            print("Total questions in topic list does not match as decided by summary... regenerating topics...")
            return False
        # focus_area_list = all_focus_skills(state.interview_topics)

        # print(f"Skill Tree List {all_skill_leaves}")

        # print(f"\nFocus Area List {focus_area_list}")
        for i in focus_area_list:
            if i not in all_skill_leaves:
                return False
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
