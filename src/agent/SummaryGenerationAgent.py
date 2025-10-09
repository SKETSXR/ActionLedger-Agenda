# =============================================================================
# Module: summary_generation_agent
# =============================================================================
# Overview
#   Production-grade LangGraph node that orchestrates a single LLM call to
#   generate a structured interview summary from three inputs:
#     - job_description
#     - skill_tree
#     - candidate_profile
#
# Responsibilities
#   - Build deterministic System+Human messages from AgentInternalState.
#   - Invoke LLM with structured output (GeneratedSummarySchema).
#   - Apply timeouts and bounded, exponential-backoff retries.
#   - Persist result to state.generated_summary.
#   - Emit operational logs (JSON file + human-readable console).
#
# Data Flow
#   StateGraph:  START ──► summary_generator (async) ──► END
#     1) Validate state fields.
#     2) Render prompt from state payloads (JSON-serialized).
#     3) Call LLM (structured output, timeout, retries).
#     4) Write GeneratedSummarySchema to state.
#
# Public API
#   - SummaryGenerationAgent.summary_generator(state) -> state
#   - SummaryGenerationAgent.get_graph(checkpointer=None) -> Compiled graph
#
# Reliability & Observability
#   - Timeout per LLM request (env-configurable).
#   - Retries with exponential backoff (env-configurable).
#   - Structured JSON logs for ingestion; concise console logs for operators.
#   - request_id attached to all LLM attempts for traceability.
#
# Configuration (Environment Variables)
#   SUMMARY_AGENT_LOG_DIR                     (default: logs)
#   SUMMARY_AGENT_LOG_FILE                    (default: summary_generation_agent.log)
#   SUMMARY_AGENT_LOG_LEVEL_CONSOLE           (default: INFO)
#   SUMMARY_AGENT_LOG_LEVEL_FILE              (default: INFO)
#   SUMMARY_AGENT_LOG_BACKUP_DAYS             (default: 365)
#   SUMMARY_AGENT_LLM_TIMEOUT_SECONDS         (default: 90)
#   SUMMARY_AGENT_LLM_RETRIES                 (default: 2)
#   SUMMARY_AGENT_LLM_RETRY_BACKOFF_SECONDS   (default: 2.5)
#
# Dependencies
#   - langgraph for orchestration
#   - langchain-core for messages & structured output
#   - pydantic models in ../schema/output_schema.py (GeneratedSummarySchema)
#   - llm client in ../model_handling.py (llm_sg)
#
# Security & Privacy
#   - Logs exclude payload bodies by default; adjust only if required.
#   - Ensure upstream redaction of sensitive data before state is populated.
#
# Usage
#   - Import and add to a larger LangGraph
#
# Notes
#   - All LLM inputs are JSON-serialized to minimize prompt drift.
#   - The LLM client is injected to keep the module testable and swappable.
# =============================================================================


import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Sequence, List

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START

from ..prompt.summary_generation_agent_prompt import SUMMARY_GENERATION_AGENT_PROMPT
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import GeneratedSummarySchema
from ..model_handling import llm_sg as _llm_client


# ==============================
# Configuration
# ==============================

@dataclass(frozen=True)
class SummaryAgentConfig:
    log_dir: str = os.getenv("SUMMARY_AGENT_LOG_DIR", "logs")
    log_file: str = os.getenv("SUMMARY_AGENT_LOG_FILE", "summary_generation_agent.log")
    log_level_console: str = os.getenv("SUMMARY_AGENT_LOG_LEVEL_CONSOLE", "INFO")
    log_level_file: str = os.getenv("SUMMARY_AGENT_LOG_LEVEL_FILE", "INFO")
    log_backup_days: int = int(os.getenv("SUMMARY_AGENT_LOG_BACKUP_DAYS", "365"))

    llm_timeout_seconds: float = float(os.getenv("SUMMARY_AGENT_LLM_TIMEOUT_SECONDS", "90"))
    llm_retries: int = int(os.getenv("SUMMARY_AGENT_LLM_RETRIES", "2"))
    llm_retry_backoff_seconds: float = float(os.getenv("SUMMARY_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))


CONFIG = SummaryAgentConfig()


# ==============================
# Logging
# ==============================

class JsonLogFormatter(logging.Formatter):
    """Emit log records as JSON for easy ingestion."""
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds") + "Z",
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key in ("request_id", "agent", "event", "node"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)


def _ensure_directory(path: str) -> None:
    """Create a directory if it does not exist (best-effort)."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def get_logger(name: str = "summary_generation_agent") -> logging.Logger:
    """Return a configured logger; idempotent (won’t duplicate handlers)."""
    logger = logging.getLogger(name)
    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter below
    _ensure_directory(CONFIG.log_dir)

    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(CONFIG.log_dir, CONFIG.log_file),
        when="midnight",
        backupCount=CONFIG.log_backup_days,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, CONFIG.log_level_file.upper(), logging.INFO))
    file_handler.setFormatter(JsonLogFormatter())

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(getattr(logging, CONFIG.log_level_console.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    logger._initialized = True  # type: ignore[attr-defined]
    return logger


LOGGER = get_logger()


# ==============================
# Prompt construction & LLM call
# ==============================

def build_prompt_messages(state: AgentInternalState) -> List[SystemMessage | HumanMessage]:
    """
    Build the System + Human messages for the LLM call.
    Uses JSON-serialized inputs to avoid formatting issues.
    """
    system_msg = SystemMessage(
        content=SUMMARY_GENERATION_AGENT_PROMPT.format(
            job_description=state.job_description.model_dump_json(),
            skill_tree=state.skill_tree.model_dump_json(),
            candidate_profile=state.candidate_profile.model_dump_json(),
        )
    )
    human_msg = HumanMessage(content="Begin the summary generation per the instructions and input payload.")
    return [system_msg, human_msg]


async def call_llm_with_retries(messages: Sequence[SystemMessage | HumanMessage], request_id: str) -> GeneratedSummarySchema:
    """
    Call the LLM with structured output and retry on error/timeout using exponential backoff.
    Raises the last exception if all retries fail.
    """
    attempt = 0
    last_exc: Optional[Exception] = None

    while attempt <= CONFIG.llm_retries:
        try:
            LOGGER.info("Calling LLM", extra={"request_id": request_id, "event": "llm_call_start"})
            coro = _llm_client.with_structured_output(
                GeneratedSummarySchema, method="function_calling"
            ).ainvoke(messages)

            result: GeneratedSummarySchema = await asyncio.wait_for(coro, timeout=CONFIG.llm_timeout_seconds)

            LOGGER.info("LLM call succeeded", extra={"request_id": request_id, "event": "llm_call_success"})
            return result

        except asyncio.TimeoutError as e:
            last_exc = e
            LOGGER.error("LLM call timed out", extra={"request_id": request_id, "event": "llm_timeout"})
        except Exception as e:
            last_exc = e
            LOGGER.error("LLM call failed", extra={"request_id": request_id, "event": "llm_error"})

        attempt += 1
        if attempt <= CONFIG.llm_retries:
            sleep_seconds = CONFIG.llm_retry_backoff_seconds * (2 ** (attempt - 1))
            await asyncio.sleep(sleep_seconds)

    assert last_exc is not None
    raise last_exc


# ==============================
# Agent
# ==============================

class SummaryGenerationAgent:
    """
    Produces a structured interview summary from:
      - job_description
      - skill_tree
      - candidate_profile

    The LLM is instructed via a SystemMessage and triggered with a HumanMessage.
    The output is coerced to GeneratedSummarySchema and stored on
    AgentInternalState.generated_summary.
    """

    llm_client = _llm_client  # kept for API symmetry/swappability

    @staticmethod
    async def summary_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Build prompt from state, invoke structured-output LLM with retries/timeouts,
        assign to state.generated_summary, and return the mutated state.
        """
        request_id = getattr(state, "request_id", None) or str(uuid.uuid4())
        node_name = "summary_generator"

        # Defensive state validation
        for field_name in ("job_description", "skill_tree", "candidate_profile"):
            if getattr(state, field_name, None) is None:
                LOGGER.error(
                    "Missing required field on state: %s",
                    field_name,
                    extra={"request_id": request_id, "node": node_name, "event": "state_validation_error"},
                )
                return state  # or raise ValueError if you prefer failing fast

        messages = build_prompt_messages(state)

        try:
            summary = await call_llm_with_retries(messages, request_id)
            state.generated_summary = summary
            LOGGER.info(
                "Summary generation completed",
                extra={"request_id": request_id, "node": node_name, "event": "summary_generated"},
            )
        except Exception:
            LOGGER.error(
                "Summary generation failed",
                extra={"request_id": request_id, "node": node_name, "event": "summary_failed"},
            )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        """Build the LangGraph for summary generation: START -> summary_generator."""
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node("summary_generator", SummaryGenerationAgent.summary_generator)
        graph_builder.add_edge(START, "summary_generator")
        return graph_builder.compile(checkpointer=checkpointer, name="Summary Generation Agent")


# ==============================
# CLI helper
# ==============================

def draw_graph_ascii() -> None:
    """CLI preview of the compiled graph."""
    try:
        graph = SummaryGenerationAgent.get_graph()
        print(graph.get_graph().draw_ascii())
    except Exception:
        logging.getLogger(__name__).exception("Unable to draw ASCII graph.")


if __name__ == "__main__":
    draw_graph_ascii()
