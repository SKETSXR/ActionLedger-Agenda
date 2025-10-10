
import os
import sys
import json
import logging
import pymongo
from typing import Any
from dotenv import dotenv_values
from pymongo.errors import ServerSelectionTimeoutError
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from logging.handlers import TimedRotatingFileHandler

from .SummaryGenerationAgent import SummaryGenerationAgent
from .TopicGenerationAgent import TopicGenerationAgent
from .DiscussionSummaryPerTopic import PerTopicDiscussionSummaryGenerationAgent
from .NodesAgent import NodesGenerationAgent
from .QABlocksAgent import QABlockGenerationAgent

from ..schema.agent_schema import AgentInternalState
from ..schema.input_schema import InputSchema
from ..schema.output_schema import OutputSchema


# ==============================
# Logging config / toggles
# ==============================

AGENT_NAME = "agenda_generation_agent"

LOG_DIR = os.getenv("AGENDA_AGENT_LOG_DIR", "logs")
LOG_LEVEL = getattr(logging, os.getenv("AGENDA_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FILE = os.getenv("AGENDA_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("AGENDA_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("AGENDA_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("AGENDA_AGENT_LOG_BACKUP_COUNT", "365"))

# Input/Output payload logging: 'off' | 'summary' | 'full'
AGENDA_IO_LOG_PAYLOAD = os.getenv("AGENDA_IO_LOG_PAYLOAD", "off").strip().lower()
# Optional soft cap for full mode
AGENDA_IO_LOG_MAX_CHARS = int(os.getenv("AGENDA_IO_LOG_MAX_CHARS", "0"))


def _build_logger() -> logging.Logger:
    logger = logging.getLogger(AGENT_NAME)
    if logger.hasHandlers():
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    os.makedirs(LOG_DIR, exist_ok=True)
    file_path = os.path.join(LOG_DIR, LOG_FILE)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOG_LEVEL)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    rotating = TimedRotatingFileHandler(
        file_path,
        when=LOG_ROTATE_WHEN,
        interval=LOG_ROTATE_INTERVAL,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    logging.raiseExceptions = False
    rotating.setLevel(LOG_LEVEL)
    rotating.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(console)
    logger.addHandler(rotating)
    return logger


LOGGER = _build_logger()


def _log_info(msg: str) -> None:
    LOGGER.info(msg)


def _log_warn(msg: str) -> None:
    LOGGER.warning(msg)


def _log_err(msg: str) -> None:
    LOGGER.error(msg)


# ==============================
# JSON helpers + gated formatters
# ==============================

def _compact_json(v: Any) -> str:
    try:
        if hasattr(v, "model_dump_json"):
            return v.model_dump_json()
        if hasattr(v, "model_dump"):
            return json.dumps(v.model_dump(), ensure_ascii=False, indent=2)
        if hasattr(v, "dict"):
            return json.dumps(v.dict(), ensure_ascii=False, indent=2)
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, indent=2)
        return str(v)
    except Exception:
        try:
            return json.dumps(str(v), ensure_ascii=False)
        except Exception:
            return "<unserializable>"


def _to_py(v: Any) -> Any:
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump()
        except Exception:
            pass
    if hasattr(v, "dict"):
        try:
            return v.dict()
        except Exception:
            pass
    return v


def _summarize_inputs(state: AgentInternalState) -> str:
    try:
        jd = _to_py(state.job_description)
        st = _to_py(state.skill_tree)
        cp = _to_py(state.candidate_profile)
        qg = _to_py(state.question_guidelines)
        parts = []
        parts.append(f"job_description={'yes' if jd else 'no'}")
        parts.append(f"skill_tree={'yes' if st else 'no'}")
        parts.append(f"candidate_profile={'yes' if cp else 'no'}")
        qg_count = 0
        try:
            qg_count = len(
                getattr(state.question_guidelines, "question_guidelines", [])
                or (qg.get("question_guidelines", []) if isinstance(qg, dict) else [])
            )
        except Exception:
            pass
        parts.append(f"question_guidelines={qg_count} types")
        return " | ".join(parts)
    except Exception:
        return "<unavailable>"


def _summarize_outputs(state: AgentInternalState) -> str:
    try:
        parts = []
        gs = state.generated_summary
        it = getattr(state.interview_topics, "interview_topics", None)
        ds = getattr(state.discussion_summary_per_topic, "discussion_topics", None)
        nds = getattr(state.nodes, "topics_with_nodes", None)
        qas = getattr(state.qa_blocks, "qa_sets", None)

        parts.append(f"summary={'yes' if gs else 'no'}")
        parts.append(f"topics={len(it) if it else 0}")
        parts.append(f"per_topic_summaries={len(ds) if ds else 0}")
        parts.append(f"topics_with_nodes={len(nds) if nds else 0}")

        qa_topics = 0
        qa_blocks_total = 0
        if qas:
            qa_topics = len(qas)
            for s in qas:
                s = _to_py(s)
                qa_blocks_total += len(s.get("qa_blocks", []))
        parts.append(f"qa_topics={qa_topics}")
        parts.append(f"qa_blocks_total={qa_blocks_total}")
        return " | ".join(parts)
    except Exception:
        return "<unavailable>"


def _gate_payload(label: str, state_like: AgentInternalState) -> str:
    """
    label: 'input' or 'output'
    state_like: AgentInternalState at that point in the pipeline
    """
    mode = AGENDA_IO_LOG_PAYLOAD
    if mode == "off":
        return f"{label}=<hidden>"
    if mode == "summary":
        return (
            f"{label}={_summarize_outputs(state_like)}"
            if label.startswith("output")
            else f"{label}={_summarize_inputs(state_like)}"
        )
    # full
    s = _compact_json(state_like)
    if AGENDA_IO_LOG_MAX_CHARS and len(s) > AGENDA_IO_LOG_MAX_CHARS:
        s = s[:AGENDA_IO_LOG_MAX_CHARS].rstrip() + "…"
    return f"{label}={s}"


# ==============================
# Agenda Generation Agent
# ==============================

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
    """

    # Environment variables loaded once at import time
    env_vars = dotenv_values()

    @staticmethod
    async def input_formatter(state: InputSchema, config: RunnableConfig) -> AgentInternalState:
        """
        Convert external InputSchema + RunnableConfig into the internal AgentInternalState.
        """
        if "MONGO_CLIENT" not in AgendaGenerationAgent.env_vars:
            raise ValueError("MONGO_CLIENT is not set")
        if "MONGO_DB" not in AgendaGenerationAgent.env_vars:
            raise ValueError("MONGO_DB is not set")
        if "MONGO_SUMMARY_COLLECTION" not in AgendaGenerationAgent.env_vars:
            raise ValueError("MONGO_SUMMARY_COLLECTION is not set")

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
            id=config["configurable"]["thread_id"],
        )

        try:
            _log_info(
                f"Agenda input prepared | thread_id={internal_state.id} | {_gate_payload('input', internal_state)}"
            )
        except Exception:
            _log_warn("Agenda input prepared (logging failed to render payload)")

        return internal_state

    @staticmethod
    async def store_inp_summary_tool(state: AgentInternalState) -> AgentInternalState:
        """
        Persist input artifacts and generated summary to MongoDB (idempotent upserts by thread_id).
        Also persists question guidelines keyed by question_type_name.
        """
        client = pymongo.MongoClient(state.mongo_client)

        jd_collection = client[state.mongo_db][state.mongo_jd_collection]
        cv_collection = client[state.mongo_db][state.mongo_cv_collection]
        skill_tree_collection = client[state.mongo_db][state.mongo_skill_tree_collection]
        summary_collection = client[state.mongo_db][state.mongo_summary_collection]
        question_guidelines_collection = client[state.mongo_db][state.mongo_question_guidelines_collection]

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

        jd["_id"] = state.id
        cv["_id"] = state.id
        skill_tree["_id"] = state.id
        summary["_id"] = state.id

        try:
            jd_collection.replace_one({"_id": state.id}, jd, upsert=True)
            cv_collection.replace_one({"_id": state.id}, cv, upsert=True)
            skill_tree_collection.replace_one({"_id": state.id}, skill_tree, upsert=True)
            summary_collection.replace_one({"_id": state.id}, summary, upsert=True)
            _log_info(f"Mongo upserts completed | thread_id={state.id}")
        except ServerSelectionTimeoutError as server_error:
            _log_err(f"Mongo server timeout | thread_id={state.id} | error={server_error}")

        for raw_item in state.question_guidelines.question_guidelines:
            item = raw_item.model_dump()
            name = str(item.get("question_type_name", "")).strip()
            text = str(item.get("question_guidelines", "")).strip()

            if not name:
                _log_warn("Skipping guideline: empty 'question_type_name'")
                continue
            if not text:
                _log_warn(f"Skipping guideline '{name}': empty 'question_guidelines'")
                continue

            doc = {
                "_id": name,
                "question_type_name": name,
                "question_guidelines": text,
            }

            try:
                question_guidelines_collection.replace_one({"_id": name}, doc, upsert=True)
            except Exception as e:
                _log_err(f"Failed to upsert guideline '{name}' | error={e}")

        client.close()
        return state

    @staticmethod
    async def output_formatter(state: AgentInternalState) -> OutputSchema:
        """
        Package final artifacts into OutputSchema and log end-to-end output.
        """
        if state.generated_summary is None:
            _log_err(f"Output formatting failed | thread_id={state.id} | reason=missing generated_summary")
            raise ValueError("Summary has not been generated")
        if state.interview_topics is None:
            _log_err(f"Output formatting failed | thread_id={state.id} | reason=missing interview_topics")
            raise ValueError("Interview topics have not been generated")

        output = OutputSchema(
            summary=state.generated_summary,
            interview_topics=state.interview_topics,
            discussion_summary_per_topic=state.discussion_summary_per_topic,
            nodes=state.nodes,
            qa_blocks=state.qa_blocks,
        )

        try:
            _log_info(f"Agenda output ready | thread_id={state.id} | {_gate_payload('output', state)}")
        except Exception:
            _log_warn("Agenda output ready (logging failed to render payload)")

        return output

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Build and compile the full Agenda Generation graph with explicit edges
        reflecting the pipeline order.
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

        # Edges
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
