# =============================================================================
# Module: agenda_generation_agent
# =============================================================================
# Purpose
#   Orchestrate the full agenda-generation pipeline by wiring together
#   downstream agents (summary, topics, per-topic summaries, nodes, QA blocks),
#   persisting artifacts to Mongo, and packaging the final OutputSchema.
#
# Pipeline (LangGraph)
#   START
#     -> input_formatter
#     -> summary_generation_agent
#     -> store_inp_summary_tool
#     -> topic_generation_agent
#     -> discussion_summary_per_topic_generator
#     -> nodes_generator
#     -> qablock_generator
#     -> output_formatter
#   END
#
# Logging
#   - Console + timed rotating file logs (env-configurable)
#   - Input/Output payload gating: off | summary | full (+optional char cap)
#
# Environment Variables
#   AGENDA_AGENT_LOG_DIR, AGENDA_AGENT_LOG_LEVEL, AGENDA_AGENT_LOG_FILE
#   AGENDA_AGENT_LOG_ROTATE_WHEN, AGENDA_AGENT_LOG_ROTATE_INTERVAL
#   AGENDA_AGENT_LOG_BACKUP_COUNT
#   AGENDA_IO_LOG_PAYLOAD (off|summary|full), AGENDA_IO_LOG_MAX_CHARS
#
# Mongo (required in .env)
#   MONGO_CLIENT, MONGO_DB, MONGO_SUMMARY_COLLECTION,
#   MONGO_JD_COLLECTION, MONGO_CV_COLLECTION,
#   MONGO_SKILL_TREE_COLLECTION, MONGO_QUESTION_GENERATION_COLLECTION
# =============================================================================

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import pymongo
from dotenv import dotenv_values
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from logging.handlers import TimedRotatingFileHandler
from pymongo.errors import ServerSelectionTimeoutError

from .DiscussionSummaryPerTopic import PerTopicDiscussionSummaryAgent
from .NodesAgent import NodesGenerationAgent
from .QABlocksAgent import QABlockGenerationAgent
from .SummaryGenerationAgent import SummaryGenerationAgent
from .TopicGenerationAgent import TopicGenerationAgent
from ..schema.agent_schema import AgentInternalState
from ..schema.input_schema import InputSchema
from ..schema.output_schema import OutputSchema


# ==============================
# Configuration
# ==============================

@dataclass(frozen=True)
class AgendaConfig:
    agent_name: str = "agenda_generation_agent"

    log_dir: str = os.getenv("AGENDA_AGENT_LOG_DIR", "logs")
    log_level: int = getattr(logging, os.getenv("AGENDA_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_file: str = os.getenv("AGENDA_AGENT_LOG_FILE", f"{agent_name}.log")
    log_rotate_when: str = os.getenv("AGENDA_AGENT_LOG_ROTATE_WHEN", "midnight")
    log_rotate_interval: int = int(os.getenv("AGENDA_AGENT_LOG_ROTATE_INTERVAL", "1"))
    log_backup_count: int = int(os.getenv("AGENDA_AGENT_LOG_BACKUP_COUNT", "365"))

    # IO payload logging
    io_log_mode: str = os.getenv("AGENDA_IO_LOG_PAYLOAD", "off").strip().lower()  # off | summary | full
    io_log_max_chars: int = int(os.getenv("AGENDA_IO_LOG_MAX_CHARS", "0"))        # soft cap for full mode


CFG = AgendaConfig()


# ==============================
# Logger
# ==============================

def _build_logger() -> logging.Logger:
    logger = logging.getLogger(CFG.agent_name)
    if logger.handlers:
        return logger

    logger.setLevel(CFG.log_level)
    logger.propagate = False

    os.makedirs(CFG.log_dir, exist_ok=True)
    path = os.path.join(CFG.log_dir, CFG.log_file)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(CFG.log_level)
    console.setFormatter(fmt)

    rotating = TimedRotatingFileHandler(
        path,
        when=CFG.log_rotate_when,
        interval=CFG.log_rotate_interval,
        backupCount=CFG.log_backup_count,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    logging.raiseExceptions = False  # don’t raise on logging I/O errors
    rotating.setLevel(CFG.log_level)
    rotating.setFormatter(fmt)

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
    """Pretty JSON for dict/list or Pydantic models; fall back to str."""
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
    """Best-effort convert model → dict for summary counting."""
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
    """Compact, safe high-signal summary of inputs."""
    try:
        jd = _to_py(state.job_description)
        st = _to_py(state.skill_tree)
        cp = _to_py(state.candidate_profile)
        qg = _to_py(state.question_guidelines)

        parts: list[str] = []
        parts.append(f"job_description={'yes' if jd else 'no'}")
        parts.append(f"skill_tree={'yes' if st else 'no'}")
        parts.append(f"candidate_profile={'yes' if cp else 'no'}")

        try:
            qg_list = getattr(state.question_guidelines, "question_guidelines", None)
            if qg_list is None and isinstance(qg, dict):
                qg_list = qg.get("question_guidelines", [])
            qg_count = len(qg_list or [])
        except Exception:
            qg_count = 0
        parts.append(f"question_guidelines={qg_count} types")

        return " | ".join(parts)
    except Exception:
        return "<unavailable>"


def _summarize_outputs(state: AgentInternalState) -> str:
    """Compact summary of produced artifacts for end-of-pipeline logging."""
    try:
        parts: list[str] = []

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
                qa_blocks_total += len((s or {}).get("qa_blocks", []))
        parts.append(f"qa_topics={qa_topics}")
        parts.append(f"qa_blocks_total={qa_blocks_total}")

        return " | ".join(parts)
    except Exception:
        return "<unavailable>"


def _gate_payload(label: str, state_like: AgentInternalState) -> str:
    """
    Gate input/output payloads for logs.
      label: 'input' or 'output'
    """
    mode = CFG.io_log_mode
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
    if CFG.io_log_max_chars and len(s) > CFG.io_log_max_chars:
        s = s[:CFG.io_log_max_chars].rstrip() + "…"
    return f"{label}={s}"


# ==============================
# Agenda Generation Agent
# ==============================

class AgendaGenerationAgent:
    """
    Orchestrates the end-to-end agenda generation:
      START → input_formatter → summary_generation_agent → store_inp_summary_tool
           → topic_generation_agent → discussion_summary_per_topic_generator
           → nodes_generator → qablock_generator → output_formatter → END
    """

    # Load once (matches existing behavior)
    env_vars = dotenv_values()

    # ---------- Node: Input formatter ----------
    @staticmethod
    async def input_formatter(state: InputSchema, config: RunnableConfig) -> AgentInternalState:
        """Convert external InputSchema → AgentInternalState; validate required Mongo settings."""
        required = ("MONGO_CLIENT", "MONGO_DB", "MONGO_SUMMARY_COLLECTION")
        for key in required:
            if key not in AgendaGenerationAgent.env_vars:
                raise ValueError(f"{key} is not set")

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
            _log_info(f"Agenda input prepared | thread_id={internal_state.id} | {_gate_payload('input', internal_state)}")
        except Exception:
            _log_warn("Agenda input prepared (logging failed to render payload)")

        return internal_state

    # ---------- Node: Persist inputs + summary ----------
    @staticmethod
    async def store_inp_summary_tool(state: AgentInternalState) -> AgentInternalState:
        """
        Persist input artifacts + generated summary to MongoDB (idempotent by thread_id).
        Also upserts question guidelines keyed by question_type_name.
        """
        client = pymongo.MongoClient(state.mongo_client)

        jd_collection = client[state.mongo_db][state.mongo_jd_collection]
        cv_collection = client[state.mongo_db][state.mongo_cv_collection]
        skill_tree_collection = client[state.mongo_db][state.mongo_skill_tree_collection]
        summary_collection = client[state.mongo_db][state.mongo_summary_collection]
        qg_collection = client[state.mongo_db][state.mongo_question_guidelines_collection]

        # Precondition checks
        if not state.candidate_profile:
            raise ValueError("`candidate profile` cannot be null")
        if not state.job_description:
            raise ValueError("`job description` cannot be null")
        if not state.skill_tree:
            raise ValueError("`skill tree` cannot be null")
        if not state.generated_summary:
            raise ValueError("`generated summary` cannot be null")

        # Prepare docs with _id = thread_id
        jd = state.job_description.model_dump()
        jd["_id"] = state.id
        cv = state.candidate_profile.model_dump()
        cv["_id"] = state.id
        st = state.skill_tree.model_dump()
        st["_id"] = state.id
        sm = state.generated_summary.model_dump()
        sm["_id"] = state.id

        try:
            jd_collection.replace_one({"_id": state.id}, jd, upsert=True)
            cv_collection.replace_one({"_id": state.id}, cv, upsert=True)
            skill_tree_collection.replace_one({"_id": state.id}, st, upsert=True)
            summary_collection.replace_one({"_id": state.id}, sm, upsert=True)
            _log_info(f"Mongo upserts completed | thread_id={state.id}")
        except ServerSelectionTimeoutError as e:
            _log_err(f"Mongo server timeout | thread_id={state.id} | error={e}")

        # Upsert question guidelines by name
        for raw in state.question_guidelines.question_guidelines:
            item = raw.model_dump()
            name = str(item.get("question_type_name", "")).strip()
            text = str(item.get("question_guidelines", "")).strip()

            if not name:
                _log_warn("Skipping guideline: empty 'question_type_name'")
                continue
            if not text:
                _log_warn(f"Skipping guideline '{name}': empty 'question_guidelines'")
                continue

            doc = {"_id": name, "question_type_name": name, "question_guidelines": text}
            try:
                qg_collection.replace_one({"_id": name}, doc, upsert=True)
            except Exception as e:
                _log_err(f"Failed to upsert guideline '{name}' | error={e}")

        client.close()
        return state

    # ---------- Node: Output formatter ----------
    @staticmethod
    async def output_formatter(state: AgentInternalState) -> OutputSchema:
        """Package final artifacts into OutputSchema; log gated output view."""
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

    # ---------- Graph builder ----------
    @staticmethod
    def get_graph(checkpointer=None):
        """
        Build the full Agenda graph with explicit node order and edges.
        """
        gb = StateGraph(
            state_schema=AgentInternalState,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )

        # Nodes
        gb.add_node("input_formatter", AgendaGenerationAgent.input_formatter, input_schema=InputSchema)
        gb.add_node("summary_generation_agent", SummaryGenerationAgent.get_graph())
        gb.add_node("store_inp_summary_tool", AgendaGenerationAgent.store_inp_summary_tool)
        gb.add_node("topic_generation_agent", TopicGenerationAgent.get_graph())
        gb.add_node("discussion_summary_per_topic_generator", PerTopicDiscussionSummaryAgent.get_graph())
        gb.add_node("nodes_generator", NodesGenerationAgent.get_graph())
        gb.add_node("qablock_generator", QABlockGenerationAgent.get_graph())
        gb.add_node("output_formatter", AgendaGenerationAgent.output_formatter)

        # Edges
        gb.add_edge(START, "input_formatter")
        gb.add_edge("input_formatter", "summary_generation_agent")
        gb.add_edge("summary_generation_agent", "store_inp_summary_tool")
        gb.add_edge("store_inp_summary_tool", "topic_generation_agent")
        gb.add_edge("topic_generation_agent", "discussion_summary_per_topic_generator")
        gb.add_edge("discussion_summary_per_topic_generator", "nodes_generator")
        gb.add_edge("nodes_generator", "qablock_generator")
        gb.add_edge("qablock_generator", "output_formatter")
        gb.add_edge("output_formatter", END)

        return gb.compile(checkpointer=checkpointer, name="Agenda Generation Agent")


if __name__ == "__main__":
    graph = AgendaGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
