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
# from ..utils import serialize_candidate_relevance


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

        # if 'MONGO_INFERRED_TOPICS_COLLECTION' not in AgendaGenerationAgent.env_vars.keys():
        #     raise ValueError("MONGO_INFERRED_TOPICS_COLLECTION is not set")

        internal_state = AgentInternalState(
            job_description=state.job_description,
            skill_tree=state.skill_tree,
            candidate_profile=state.candidate_profile,
            question_guidelines=state.question_guidelines,
            mongo_client=AgendaGenerationAgent.env_vars['MONGO_CLIENT'],
            mongo_db=AgendaGenerationAgent.env_vars['MONGO_DB'],
            # mongo_inferred_topics_collection=AgendaGenerationAgent.env_vars['MONGO_INFERRED_TOPICS_COLLECTION'],
            mongo_summary_collection=AgendaGenerationAgent.env_vars['MONGO_SUMMARY_COLLECTION'],
            mongo_jd_collection=AgendaGenerationAgent.env_vars['MONGO_JD_COLLECTION'],
            mongo_cv_collection=AgendaGenerationAgent.env_vars['MONGO_CV_COLLECTION'],
            mongo_skill_tree_collection=AgendaGenerationAgent.env_vars['MONGO_SKILL_TREE_COLLECTION'],
            mongo_question_guidelines_collection=AgendaGenerationAgent.env_vars['MONGO_QUESTION_GENERATION_COLLECTION'],

            id=config['configurable']['thread_id']
        )
        return internal_state


    # @staticmethod
    # async def store_inferred_topics_tool(state: AgentInternalState) -> AgentInternalState:
    #     client = pymongo.MongoClient(state.mongo_client)
    #     inferred_topics_collection = client[state.mongo_db][state.mongo_inferred_topics_collection]

    #     if not state.inferred_interview_topics:
    #         raise ValueError("`inferred_interview_topics` cannot be null")

    #     inferred_topics = state.inferred_interview_topics.model_dump()
    #     inferred_topics['_id'] = state.id

    #     try:
    #         inferred_topics_collection.insert_one(inferred_topics)
    #     except ServerSelectionTimeoutError as server_error:
    #         print(f"Could not communicate with MongoDB server, thus inferred interview topics were not stored.: {server_error}") # TODO: In production, use logs instead of printing

    #     client.close()
    #     return state


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
        # question_guidelines = state.question_guidelines.model_dump()
        jd['_id'] = state.id
        cv['_id'] = state.id
        skill_tree['_id'] = state.id
        summary['_id'] = state.id

        try:

            jd_collection.insert_one(jd)
            cv_collection.insert_one(cv)
            skill_tree_collection.insert_one(skill_tree)
            summary_collection.insert_one(summary)
            # db = client[state.mongo_db]
            # collections = db.list_collection_names()
            # if "question_guidelines" not in collections:
            #     question_guidelines_collection.insert_one(question_guidelines)
        except ServerSelectionTimeoutError as server_error:
            print(f"Could not communicate with MongoDB server, thus summary was not stored.: {server_error}")
        
        # # ----- question_guidelines upsert(s) -----
        # # Pydantic will coerce your dict input; use the INSTANCE on state
        # qg_model = state.question_guidelines
        # guideline_text = qg_model.question_guidelines  # str
        # type_names = qg_model.question_type_name       # list[str]

        # # Optional: make sure we have a unique index on _id (implicit) or on question_type_name if you prefer that field.
        # # qg_collection.create_index("question_type_name", unique=True)  # only if you use that as key instead of _id

        # for name in type_names:
        #     # one document per question type
        #     doc = {
        #         "_id": str(name),  # using the type name as the stable primary key
        #         "question_type_name": name,
        #         "question_guidelines": guideline_text
        #         # "updated_at": datetime.utcnow()
        #     }
        #     try:
        #         # upsert ensures idempotency
        #         question_guidelines_collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        #         # print(f" Upserted guideline for '{name}'")
        #     except Exception as e:
        #         # print(f" Guideline '{name}' already exists with the same _id. Skipping.")
        #         print(f"Error upserting the guidelines into the database {e}")
        #     # except PyMongoError as e:
        #     #     print(f" Failed to upsert guideline '{name}': {e}")

        # ---- Upsert each guideline item (list[QuestionGuidelineItem]) ----
        # Optional (safe to run repeatedly): dedicated unique index if you ever switch off _id as key
        # qg_collection.create_index("question_type_name", unique=True, name="uq_qg_type")

        for raw_item in state.question_guidelines.question_guidelines:   # list of models or dicts
            # item = _as_dict(raw_item)
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
        # graph_builder.add_node("store_inp_summary_tool", AgendaGenerationAgent.store_summary_tool)
        graph_builder.add_node("store_inp_summary_tool", AgendaGenerationAgent.store_inp_summary_tool)
        # graph_builder.add_node("interview_topics_storage_tool", AgendaGenerationAgent.store_inferred_topics_tool)
        graph_builder.add_node("output_formatter", AgendaGenerationAgent.output_formatter)

        graph_builder.add_edge(START, "input_formatter")
        graph_builder.add_edge("input_formatter", "summary_generation_agent")
        graph_builder.add_edge("summary_generation_agent", "store_inp_summary_tool")
        graph_builder.add_edge("store_inp_summary_tool", "topic_generation_agent")
        graph_builder.add_edge("topic_generation_agent", "discussion_summary_per_topic_generator")
        graph_builder.add_edge("discussion_summary_per_topic_generator", "nodes_generator")
        graph_builder.add_edge("nodes_generator", "qablock_generator")
        graph_builder.add_edge("qablock_generator", "output_formatter")
        # graph_builder.add_edge("discussion_summary_per_topic_generator", "output_formatter")
        # graph_builder.add_edge("store_inp_summary_tool", "output_formatter")
        graph_builder.add_edge("output_formatter", END)

        graph = graph_builder.compile(checkpointer=checkpointer, name="Agenda Generation Agent")

        return graph


if __name__ == "__main__":
    graph = AgendaGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
