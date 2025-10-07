import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Sequence

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START

from ..prompt.summary_generation_agent_prompt import SUMMARY_GENERATION_AGENT_PROMPT
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import GeneratedSummarySchema
from ..model_handling import llm_sg as _llm_client


# ==============================
# Config
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
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds") + "Z",
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Include extras if present
        for key in ("request_id", "agent", "event", "node"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)

def _ensure_log_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # Fall back to current directory if we cannot make log dir.
        pass

def get_logger(name: str = "summary_generation_agent") -> logging.Logger:
    """
    Returns a configured logger. Safe to call multiple times without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter levels
    _ensure_log_dir(CONFIG.log_dir)

    # File handler (rotates daily)
    fh = TimedRotatingFileHandler(
        filename=os.path.join(CONFIG.log_dir, CONFIG.log_file),
        when="midnight",
        backupCount=CONFIG.log_backup_days,
        encoding="utf-8"
    )
    fh.setLevel(getattr(logging, CONFIG.log_level_file.upper(), logging.INFO))
    fh.setFormatter(JsonLogFormatter())

    # Console handler (human-readable)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, CONFIG.log_level_console.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    logger._initialized = True  # type: ignore[attr-defined]
    return logger

logger = get_logger()


# ==============================
# Utilities
# ==============================

def _build_messages(state: AgentInternalState) -> Sequence:
    """
    Builds the System + Human messages for the LLM call.
    Uses explicit JSON dumps to guarantee valid strings.
    """
    system = SystemMessage(
        content=SUMMARY_GENERATION_AGENT_PROMPT.format(
            job_description=state.job_description.model_dump_json(),
            skill_tree=state.skill_tree.model_dump_json(),
            candidate_profile=state.candidate_profile.model_dump_json(),
        )
    )
    trigger = HumanMessage(
        content="Begin the summary generation per the instructions and input payload."
    )
    return [system, trigger]


async def _call_llm_with_retries(messages: Sequence, request_id: str) -> GeneratedSummarySchema:
    """
    Calls the LLM with structured output, applying timeout and simple exponential backoff.
    Raises the last exception if all retries fail.
    """
    attempt = 0
    last_exc: Optional[Exception] = None

    while attempt <= CONFIG.llm_retries:
        try:
            logger.info(
                "Calling LLM", extra={"request_id": request_id, "event": "llm_call_start"}
            )
            coro = _llm_client.with_structured_output(
                GeneratedSummarySchema, method="function_calling"
            ).ainvoke(messages)

            result: GeneratedSummarySchema = await asyncio.wait_for(
                coro, timeout=CONFIG.llm_timeout_seconds
            )

            logger.info(
                "LLM call succeeded",
                extra={"request_id": request_id, "event": "llm_call_success"}
            )
            return result

        except asyncio.TimeoutError as te:
            last_exc = te
            logger.error(
                "LLM call timed out",
                extra={"request_id": request_id, "event": "llm_timeout"}
            )
        except Exception as e:
            last_exc = e
            logger.error(
                "LLM call failed",
                extra={"request_id": request_id, "event": "llm_error"}
            )

        attempt += 1
        if attempt <= CONFIG.llm_retries:
            backoff = CONFIG.llm_retry_backoff_seconds * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

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

    The LLM is instructed via a SystemMessage and triggered with a minimal HumanMessage.
    Output is coerced to GeneratedSummarySchema and stored on AgentInternalState.generated_summary.
    """

    # Share one llm client across the class (taken via import), but keep interface swappable if needed.
    llm_client = _llm_client

    @staticmethod
    async def summary_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Build prompt from state, invoke structured-output LLM with retries/timeouts,
        assign to state.generated_summary, and return the mutated state.
        """
        request_id = getattr(state, "request_id", None) or str(uuid.uuid4())
        node_name = "summary_generator"

        # Validate state fields (defensive)
        for attr in ("job_description", "skill_tree", "candidate_profile"):
            if getattr(state, attr, None) is None:
                logger.error(
                    "Missing required field on state: %s", attr,
                    extra={"request_id": request_id, "node": node_name, "event": "state_validation_error"}
                )
                return state  # or raise ValueError to fail fast

        messages = _build_messages(state)

        try:
            summary: GeneratedSummarySchema = await _call_llm_with_retries(messages, request_id)
            state.generated_summary = summary
            logger.info(
                "Summary generation completed",
                extra={
                    "request_id": request_id,
                    "node": node_name,
                    "event": "summary_generated",
                },
            )
        except Exception as e:
            # If required in your schema, you could attach an error object here.
            logger.error(
                "Summary generation failed",
                extra={"request_id": request_id, "node": node_name, "event": "summary_failed"}
            )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Build the LangGraph for summary generation: START -> summary_generator
        """
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node("summary_generator", SummaryGenerationAgent.summary_generator)
        graph_builder.add_edge(START, "summary_generator")
        return graph_builder.compile(checkpointer=checkpointer, name="Summary Generation Agent")


# ==============================
# CLI helper (optional)
# ==============================

def _draw_graph_ascii() -> None:
    """
    Safe CLI preview of the compiled graph.
    """
    try:
        graph = SummaryGenerationAgent.get_graph()
        print(graph.get_graph().draw_ascii())
    except Exception:
        # Avoid crashing on import-time issues in CLI contexts.
        logging.getLogger(__name__).exception("Unable to draw ASCII graph.")


if __name__ == "__main__":
    _draw_graph_ascii()
