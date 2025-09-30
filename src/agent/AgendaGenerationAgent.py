from dotenv import dotenv_values
import pymongo
from pymongo.errors import ServerSelectionTimeoutError
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from .SummaryGenerationAgent import SummaryGenerationAgent
from .TopicGenerationAgent import TopicGenerationAgent
from .DiscussionSummaryPerTopic import PerTopicDiscussionSummaryGenerationAgent
from .NodesAgent import NodesGenerationAgent
from .QABlocksAgent import QABlockGenerationAgent

from ..schema.agent_schema import AgentInternalState
from ..schema.input_schema import InputSchema
from ..schema.output_schema import OutputSchema


class AgendaGenerationAgent:
    """
    Orchestrates the full agenda-generation pipeline:

    START
      -> input_formatter
      -> summary_generation_agent
      -> store_inp_summary_tool (persist inputs + summary to MongoDB)
      -> topic_generation_agent
      -> discussion_summary_per_topic_generator
      -> nodes_generator
      -> qablock_generator
      -> output_formatter
    END

    Notes:
    - Minimal validation guards ensure required state is present.
    - Mongo persistence uses replace_one(upsert=True) with _id = thread_id for idempotency.
    """

    # Environment variables loaded once at import time
    env_vars = dotenv_values()

    @staticmethod
    async def input_formatter(state: InputSchema, config: RunnableConfig) -> AgentInternalState:
        """
        Convert external InputSchema + RunnableConfig into the internal AgentInternalState
        used across the pipeline. Validates the presence of required Mongo settings.

        Raises:
            ValueError: if required Mongo env vars are missing.
        """
        if "MONGO_CLIENT" not in AgendaGenerationAgent.env_vars:
            raise ValueError("MONGO_CLIENT is not set")
        if "MONGO_DB" not in AgendaGenerationAgent.env_vars:
            raise ValueError("MONGO_DB is not set")
        if "MONGO_SUMMARY_COLLECTION" not in AgendaGenerationAgent.env_vars:
            raise ValueError("MONGO_SUMMARY_COLLECTION is not set")

        # Build the internal working state used by downstream agents
        internal_state = AgentInternalState(
            job_description=state.job_description,
            skill_tree=state.skill_tree,
            candidate_profile=state.candidate_profile,
            question_guidelines=state.question_guidelines,

            mongo_client=AgendaGenerationAgent.env_vars["MONGO_CLIENT"],
            mongo_db=AgendaGenerationAgent.env_vars["MONGO_DB"],
            mongo_summary_collection=AgendaGenerationAgent.env_vars["MONGO_SUMMARY_COLLECTION"],
            mongo_jd_collection=AgendaGenerationAgent.env_vars["MONGO_JD_COLLECTION"],
            mongo_cv_collection=AgendaGenerationAgent.env_vars["MONGO_CV_COLLECTION"],
            mongo_skill_tree_collection=AgendaGenerationAgent.env_vars["MONGO_SKILL_TREE_COLLECTION"],
            mongo_question_guidelines_collection=AgendaGenerationAgent.env_vars["MONGO_QUESTION_GENERATION_COLLECTION"],

            # Thread id from LangGraph config, used as stable _id for Mongo documents
            id=config["configurable"]["thread_id"],
        )
        return internal_state

    @staticmethod
    async def store_inp_summary_tool(state: AgentInternalState) -> AgentInternalState:
        """
        Persist the input artifacts and generated summary to MongoDB using the
        thread_id as the document _id for idempotent upserts.

        Also persists question guidelines keyed by question_type_name.

        Raises:
            ValueError: if required pieces of state are missing.
        """
        client = pymongo.MongoClient(state.mongo_client)

        jd_collection = client[state.mongo_db][state.mongo_jd_collection]
        cv_collection = client[state.mongo_db][state.mongo_cv_collection]
        skill_tree_collection = client[state.mongo_db][state.mongo_skill_tree_collection]
        summary_collection = client[state.mongo_db][state.mongo_summary_collection]
        question_guidelines_collection = client[state.mongo_db][state.mongo_question_guidelines_collection]

        # Required data guards
        if not state.candidate_profile:
            raise ValueError("`candidate profile` cannot be null")
        if not state.job_description:
            raise ValueError("`job description` cannot be null")
        if not state.skill_tree:
            raise ValueError("`skill tree` cannot be null")
        if not state.generated_summary:
            raise ValueError("`generated summary` cannot be null")

        # Prepare documents (attach _id = thread id for safe upsert)
        jd = state.job_description.model_dump()
        cv = state.candidate_profile.model_dump()
        skill_tree = state.skill_tree.model_dump()
        summary = state.generated_summary.model_dump()

        jd["_id"] = state.id
        cv["_id"] = state.id
        skill_tree["_id"] = state.id
        summary["_id"] = state.id

        try:
            # Upsert each document for idempotency
            jd_collection.replace_one({"_id": state.id}, jd, upsert=True)
            cv_collection.replace_one({"_id": state.id}, cv, upsert=True)
            skill_tree_collection.replace_one({"_id": state.id}, skill_tree, upsert=True)
            summary_collection.replace_one({"_id": state.id}, summary, upsert=True)
        except ServerSelectionTimeoutError as server_error:
            # Kept as print to preserve behavior; use a logger in real deployments
            print(f"Could not communicate with MongoDB server, thus summary was not stored.: {server_error}")

        # Persist question guidelines, keyed by question_type_name
        for raw_item in state.question_guidelines.question_guidelines:
            item = raw_item.model_dump()

            name = str(item.get("question_type_name", "")).strip()
            text = str(item.get("question_guidelines", "")).strip()

            if not name:
                print(" Skipping a guideline with empty 'question_type_name'.")
                continue
            if not text:
                print(f" Skipping '{name}' because 'question_guidelines' is empty.")
                continue

            doc = {
                "_id": name,  # stable primary key
                "question_type_name": name,
                "question_guidelines": text,
            }

            try:
                question_guidelines_collection.replace_one({"_id": name}, doc, upsert=True)
            except Exception as e:
                print(f" Failed to upsert guideline: {e}")

        client.close()
        return state

    @staticmethod
    async def output_formatter(state: AgentInternalState) -> OutputSchema:
        """
        Package the final artifacts from internal state into OutputSchema.

        Raises:
            ValueError: if essential outputs are missing.
        """
        if state.generated_summary is None:
            raise ValueError("Summary has not been generated")
        if state.interview_topics is None:
            raise ValueError("Interview topics have not been generated")

        output = OutputSchema(
            summary=state.generated_summary,
            interview_topics=state.interview_topics,
            discussion_summary_per_topic=state.discussion_summary_per_topic,
            nodes=state.nodes,
            qa_blocks=state.qa_blocks,
        )
        return output

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Build and compile the full Agenda Generation graph with explicit
        node-to-node edges reflecting the pipeline order.
        """
        graph_builder = StateGraph(
            state_schema=AgentInternalState,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )

        # Nodes
        graph_builder.add_node("input_formatter", AgendaGenerationAgent.input_formatter, input_schema=InputSchema)
        graph_builder.add_node("summary_generation_agent", SummaryGenerationAgent.get_graph())
        graph_builder.add_node("topic_generation_agent", TopicGenerationAgent.get_graph())
        graph_builder.add_node(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.get_graph(),
        )
        graph_builder.add_node("nodes_generator", NodesGenerationAgent.get_graph())
        graph_builder.add_node("qablock_generator", QABlockGenerationAgent.get_graph())
        graph_builder.add_node("store_inp_summary_tool", AgendaGenerationAgent.store_inp_summary_tool)
        graph_builder.add_node("output_formatter", AgendaGenerationAgent.output_formatter)

        # Edges reflecting the intended control flow
        graph_builder.add_edge(START, "input_formatter")
        graph_builder.add_edge("input_formatter", "summary_generation_agent")
        graph_builder.add_edge("summary_generation_agent", "store_inp_summary_tool")
        graph_builder.add_edge("store_inp_summary_tool", "topic_generation_agent")
        graph_builder.add_edge("topic_generation_agent", "discussion_summary_per_topic_generator")
        graph_builder.add_edge("discussion_summary_per_topic_generator", "nodes_generator")
        graph_builder.add_edge("nodes_generator", "qablock_generator")
        graph_builder.add_edge("qablock_generator", "output_formatter")
        graph_builder.add_edge("output_formatter", END)

        return graph_builder.compile(checkpointer=checkpointer, name="Agenda Generation Agent")


if __name__ == "__main__":
    graph = AgendaGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
