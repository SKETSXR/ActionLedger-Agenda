from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import GeneratedSummarySchema, CollectiveInterviewTopicSchema
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from ..model_handling import llm_tg

set_llm_cache(InMemoryCache())


class TopicGenerationAgent:

    llm_tg = llm_tg

    # @staticmethod
    # async def cleanup_internal_state(state: AgentInternalState) -> AgentInternalState:
    #     state.messages = [
    #         RemoveMessage(id=str(m.id)) for m in state.messages
    #     ]
    #     return state

    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        # response = await TopicGenerationAgent.llm_tg \
        # .with_structured_output(CollectiveInterviewTopicSchema) \
        # .ainvoke(
        #     [
        #         SystemMessage(
        #             content=TOPIC_GENERATION_AGENT_PROMPT.format(
        #                 # job_description_schema=state.job_description.model_json_schema(),
        #                 # candidate_profile_schema=state.candidate_profile.model_json_schema(),
        #                 # skill_tree_schema=state.skill_tree.model_json_schema(),
        #                 # generated_summary_schema=GeneratedSummarySchema.model_json_schema(),
        #                 # inferred_interview_topics_schema=state.interview_topics,
        #                 # job_description=state.job_description.model_dump_json(),
        #                 # skill_tree=state.skill_tree.model_dump_json(),
        #                 # candidate_profile=state.candidate_profile.model_dump_json(),
        #                 generated_summary=state.generated_summary.model_dump_json(),
        #                 # inferred_interview_topics=state.inferred_interview_topics.model_dump_json() if state.inferred_interview_topics else ""
        #             )
        #         ),
        #         state.messages[-1] if len(state.messages) else ""
        #     ]
        # )
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
    def get_graph(checkpointer=None):

        graph_builder = StateGraph(
            state_schema=AgentInternalState
        )

        # graph_builder.add_node("cleanup_internal_state", TopicGenerationAgent.cleanup_internal_state)
        graph_builder.add_node("topic_generator", TopicGenerationAgent.topic_generator)

        # graph_builder.add_edge(START, "cleanup_internal_state")
        graph_builder.add_edge(START, "topic_generator")
        # graph_builder.add_edge("cleanup_internal_state", "topic_generator")
       
        graph = graph_builder.compile(checkpointer=checkpointer, name="Topic Generation Agent")

        return graph


if __name__ == "__main__":
    graph = TopicGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
