from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START

from ..prompt.summary_generation_agent_prompt import SUMMARY_GENERATION_AGENT_PROMPT
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import GeneratedSummarySchema
from ..model_handling import llm_sg


class SummaryGenerationAgent:
    """
    Produces a structured interview summary from:
      - job_description
      - skill_tree
      - candidate_profile

    The LLM is instructed via a SystemMessage and triggered with a minimal
    HumanMessage. Output is coerced to GeneratedSummarySchema.
    """

    # The class attribute for model to use the same client.
    llm_sg = llm_sg

    @staticmethod
    async def summary_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Build the prompt from state, invoke the structured-output LLM,
        and store the result on the state.
        """
        system_message = SystemMessage(
            content=SUMMARY_GENERATION_AGENT_PROMPT.format(
                job_description=state.job_description.model_dump_json(),
                skill_tree=state.skill_tree.model_dump_json(),
                candidate_profile=state.candidate_profile.model_dump_json(),
            )
        )

        # Minimal user turn to trigger the model against the instructions.
        trigger_message = HumanMessage(
            content="Based on the provided instructions please start the process"
        )

        summary = (
            SummaryGenerationAgent.llm_sg.with_structured_output(
                GeneratedSummarySchema, method="function_calling"
            )
            .ainvoke([system_message, trigger_message])
        )

        # Await the coroutine and assign to state.
        summary = await summary
        state.generated_summary = summary
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Build the LangGraph's graph for summary generation:
        START -> summary_generator
        """
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node("summary_generator", SummaryGenerationAgent.summary_generator)
        graph_builder.add_edge(START, "summary_generator")
        return graph_builder.compile(checkpointer=checkpointer, name="Summary Generation Agent")


if __name__ == "__main__":
    graph = SummaryGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
