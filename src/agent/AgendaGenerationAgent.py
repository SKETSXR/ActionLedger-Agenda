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

    env_vars = dotenv_values()

    @staticmethod
    async def input_formatter(state: InputSchema, config: RunnableConfig) -> AgentInternalState:

        if 'MONGO_CLIENT' not in AgendaGenerationAgent.env_vars.keys():
            raise ValueError("MONGO_CLIENT is not set")

        if 'MONGO_DB' not in AgendaGenerationAgent.env_vars.keys():
            raise ValueError("MONGO_DB is not set")

        if 'MONGO_SUMMARY_COLLECTION' not in AgendaGenerationAgent.env_vars.keys():
            raise ValueError("MONGO_SUMMARY_COLLECTION is not set")

        internal_state = AgentInternalState(
            job_description=state.job_description,
            skill_tree=state.skill_tree,
            candidate_profile=state.candidate_profile,
            question_guidelines=state.question_guidelines,
            mongo_client=AgendaGenerationAgent.env_vars['MONGO_CLIENT'],
            mongo_db=AgendaGenerationAgent.env_vars['MONGO_DB'],
            mongo_summary_collection=AgendaGenerationAgent.env_vars['MONGO_SUMMARY_COLLECTION'],
            mongo_jd_collection=AgendaGenerationAgent.env_vars['MONGO_JD_COLLECTION'],
            mongo_cv_collection=AgendaGenerationAgent.env_vars['MONGO_CV_COLLECTION'],
            mongo_skill_tree_collection=AgendaGenerationAgent.env_vars['MONGO_SKILL_TREE_COLLECTION'],
            mongo_question_guidelines_collection=AgendaGenerationAgent.env_vars['MONGO_QUESTION_GENERATION_COLLECTION'],

            id=config['configurable']['thread_id']
        )
        return internal_state

    @staticmethod
    async def store_inp_summary_tool(state: AgentInternalState) -> AgentInternalState:
        client = pymongo.MongoClient(state.mongo_client)
        jd_collection = client[state.mongo_db][state.mongo_jd_collection]
        cv_collection = client[state.mongo_db][state.mongo_cv_collection]
        skill_tree_collection = client[state.mongo_db][state.mongo_skill_tree_collection]
        summary_collection = client[state.mongo_db][state.mongo_summary_collection]
        question_guidelines_collection=client[state.mongo_db][state.mongo_question_guidelines_collection]

        if not state.candidate_profile:
            raise ValueError("`candidate profile` cannot be null")
        if not state.job_description:
            raise ValueError("`job description` cannot be null")
        if not state.skill_tree:
            raise ValueError("`skill tree` cannot be null")
        if not state.generated_summary:
            raise ValueError("`generated summary` cannot be null")

        jd = state.job_description.model_dump()
        cv = state.candidate_profile.model_dump()
        skill_tree = state.skill_tree.model_dump()
        summary = state.generated_summary.model_dump()
        jd['_id'] = state.id
        cv['_id'] = state.id
        skill_tree['_id'] = state.id
        summary['_id'] = state.id

        try:

            # jd_collection.insert_one(jd)
            # cv_collection.insert_one(cv)
            # skill_tree_collection.insert_one(skill_tree)
            # summary_collection.insert_one(summary)
            jd_collection.replace_one({"_id": state.id}, jd, upsert=True)
            cv_collection.replace_one({"_id": state.id}, cv, upsert=True)
            skill_tree_collection.replace_one({"_id": state.id}, skill_tree, upsert=True)
            summary_collection.replace_one({"_id": state.id}, summary, upsert=True)

        except ServerSelectionTimeoutError as server_error:
            print(f"Could not communicate with MongoDB server, thus summary was not stored.: {server_error}")

        for raw_item in state.question_guidelines.question_guidelines:   # list of models or dicts
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
                "_id": name,  # use question_type_name as stable primary key
                "question_type_name": name,
                "question_guidelines": text,
            }

            try:
                question_guidelines_collection.replace_one({"_id": name}, doc, upsert=True)
                # print(f" Upserted guideline for '{name}'")
            except Exception as e:
                print(f" Failed to upsert guideline: {e}")
        client.close()
        return state

    @staticmethod
    async def output_formatter(state: AgentInternalState) -> OutputSchema:
        if state.generated_summary is None:
            raise ValueError("Summary has not been generated")

        if state.interview_topics is None:
            raise ValueError("Interview topics have not been generated")

        output = OutputSchema(
            summary=state.generated_summary,
            interview_topics=state.interview_topics,
            discussion_summary_per_topic=state.discussion_summary_per_topic,
            nodes=state.nodes,
            qa_blocks=state.qa_blocks
        )
        return output

    @staticmethod
    def get_graph(checkpointer=None):

        graph_builder = StateGraph(
            state_schema=AgentInternalState,
            input_schema=InputSchema,
            output_schema=OutputSchema
        )

        graph_builder.add_node("input_formatter", AgendaGenerationAgent.input_formatter, input_schema=InputSchema)
        graph_builder.add_node("summary_generation_agent", SummaryGenerationAgent.get_graph())
        graph_builder.add_node("topic_generation_agent", TopicGenerationAgent.get_graph())
        graph_builder.add_node("discussion_summary_per_topic_generator", PerTopicDiscussionSummaryGenerationAgent.get_graph())
        graph_builder.add_node("nodes_generator", NodesGenerationAgent.get_graph())
        graph_builder.add_node("qablock_generator", QABlockGenerationAgent.get_graph())
        graph_builder.add_node("store_inp_summary_tool", AgendaGenerationAgent.store_inp_summary_tool)
        graph_builder.add_node("output_formatter", AgendaGenerationAgent.output_formatter)

        graph_builder.add_edge(START, "input_formatter")
        graph_builder.add_edge("input_formatter", "summary_generation_agent")
        graph_builder.add_edge("summary_generation_agent", "store_inp_summary_tool")
        graph_builder.add_edge("store_inp_summary_tool", "topic_generation_agent")
        graph_builder.add_edge("topic_generation_agent", "discussion_summary_per_topic_generator")
        graph_builder.add_edge("discussion_summary_per_topic_generator", "nodes_generator")
        graph_builder.add_edge("nodes_generator", "qablock_generator")
        graph_builder.add_edge("qablock_generator", "output_formatter")
        graph_builder.add_edge("output_formatter", END)

        graph = graph_builder.compile(checkpointer=checkpointer, name="Agenda Generation Agent")

        return graph


if __name__ == "__main__":
    graph = AgendaGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
