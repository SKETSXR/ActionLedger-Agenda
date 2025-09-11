from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from ..prompt.summary_generation_agent_prompt import SUMMARY_GENERATION_AGENT_PROMPT
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import GeneratedSummarySchema
from ..model_handling import llm_sg

set_llm_cache(InMemoryCache())


class SummaryGenerationAgent:

    llm_sg = llm_sg

    @staticmethod
    async def summary_generator(state: AgentInternalState) -> AgentInternalState:
        summary = await SummaryGenerationAgent.llm_sg \
        .with_structured_output(GeneratedSummarySchema, method="function_calling") \
        .ainvoke(
            [
                SystemMessage(
                    content=SUMMARY_GENERATION_AGENT_PROMPT.format(
                        job_description=state.job_description.model_dump_json(),
                        skill_tree=state.skill_tree.model_dump_json(),
                        candidate_profile=state.candidate_profile.model_dump_json(),
                        # generated_summary=state.generated_summary.model_dump_json() if state.generated_summary else ""
                    )
                ),
                state.messages[-1] if len(state.messages) else ""
            ]
        )
        state.generated_summary = summary
        return state

    @staticmethod
    def get_graph(checkpointer=None):

        graph_builder = StateGraph(state_schema=AgentInternalState)

        graph_builder.add_node("summary_generator", SummaryGenerationAgent.summary_generator)

        graph_builder.add_edge(START, "summary_generator")

        graph = graph_builder.compile(checkpointer=checkpointer, name="Summary Generation Agent")
        return graph


if __name__ == "__main__":
    graph = SummaryGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
