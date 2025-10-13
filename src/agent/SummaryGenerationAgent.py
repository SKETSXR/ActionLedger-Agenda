
# =============================================================================
# Module: summary_generation_agent
# =============================================================================
# Purpose
#   Orchestrate one LLM call to produce a structured interview summary from:
#     • job_description
#     • skill_tree
#     • candidate_profile
#
# Responsibilities
#   • Build deterministic System + Human messages from AgentInternalState.
#   • Invoke LLM with structured output (GeneratedSummarySchema).
#   • Enforce timeout and bounded exponential-backoff retries.
#   • Store the result in state.generated_summary.
#   • Emit JSON-style logs to file and concise logs to console.
#
# Data Flow
#   START ──► summary_generator (async) ──► END
#     1) Validate required state fields.
#     2) Render prompt with JSON-serialized inputs.
#     3) Call LLM (structured output, timeout, retries).
#     4) Write GeneratedSummarySchema to state.
#
# Interface
#   • SummaryGenerationAgent.summary_generator(state) -> state
#   • SummaryGenerationAgent.get_graph(checkpointer=None) -> Compiled graph
#
# Config (via environment variables)
#   SUMMARY_AGENT_LOG_DIR                     (default: logs)
#   SUMMARY_AGENT_LOG_FILE                    (default: summary_generation_agent.log)
#   SUMMARY_AGENT_LOG_LEVEL_CONSOLE           (default: INFO)
#   SUMMARY_AGENT_LOG_LEVEL_FILE              (default: INFO)
#   SUMMARY_AGENT_LOG_BACKUP_DAYS             (default: 365)
#   SUMMARY_AGENT_LLM_TIMEOUT_SECONDS         (default: 90)
#   SUMMARY_AGENT_LLM_RETRIES                 (default: 2)
#   SUMMARY_AGENT_LLM_RETRY_BACKOFF_SECONDS   (default: 2.5)
#   SUMMARY_AGENT_RESULT_LOG_PAYLOAD          off | summary | full   (default: off)
#
# Notes
#   • Inputs are JSON-serialized to reduce prompt drift.
#   • LLM client is injected (../model_handling.llm_sg) for testability.
# =============================================================================

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from typing import List, Optional, Sequence, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph

from ..model_handling import llm_sg as _llm_client
from ..prompt.summary_generation_agent_prompt import SUMMARY_GENERATION_AGENT_PROMPT
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import GeneratedSummarySchema


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

    # Controls how much of the model output is logged
    result_log_payload: str = os.getenv("SUMMARY_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()


CONFIG = SummaryAgentConfig()


# ==============================
# Logging
# ==============================

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # Best-effort only; avoid crashing on log-dir issues
        pass


def _get_logger(name: str = "summary_generation_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if getattr(logger, "_initialized", False):
        return logger

    logger.setLevel(logging.DEBUG)
    _ensure_dir(CONFIG.log_dir)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(CONFIG.log_dir, CONFIG.log_file),
        when="midnight",
        interval=1,
        backupCount=CONFIG.log_backup_days,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    file_handler.setLevel(getattr(logging, CONFIG.log_level_file.upper(), logging.INFO))
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(getattr(logging, CONFIG.log_level_console.upper(), logging.INFO))
    console_handler.setFormatter(fmt)

    logging.raiseExceptions = False  # Never crash on logging errors in production

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    logger._initialized = True  # type: ignore[attr-defined]
    return logger


LOGGER = _get_logger()


# ==============================
# Serialization helpers (for safe logging)
# ==============================

def _pydantic_to_dict(obj) -> object:
    """Convert Pydantic v1/v2 models to plain objects; otherwise return as is."""
    for attr in ("model_dump_json", "json"):
        if hasattr(obj, attr):
            try:
                return json.loads(getattr(obj, attr)())
            except Exception:
                pass
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return obj


def _to_json_str(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            return "<unserializable>"


def _summarize_output(payload: object) -> str:
    """
    Produce a short, schema-agnostic description: top-level keys and common fields.
    """
    try:
        data = _pydantic_to_dict(payload)
        if isinstance(data, dict):
            keys = list(data.keys())
            parts = [f"keys={keys[:8]}{'...' if len(keys) > 8 else ''}"]
            for k in ("total_questions", "topic_count", "topics", "interview_topics"):
                if k in data:
                    v = data[k]
                    parts.append(f"{k}.len={len(v) if isinstance(v, list) else v}")
            return "; ".join(parts)
        if isinstance(data, list):
            return f"list(len={len(data)})"
        return type(data).__name__
    except Exception:
        return "<unavailable>"


def _render_for_log(payload: object) -> str:
    mode = CONFIG.result_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        return _summarize_output(payload)
    # "full"
    return _to_json_str(_pydantic_to_dict(payload))


# ==============================
# Prompt creation & LLM call
# ==============================

Msg = Union[SystemMessage, HumanMessage]


def build_messages(state: AgentInternalState) -> List[Msg]:
    """
    Build deterministic System + Human messages from state.
    Inputs are JSON-serialized to keep formatting stable across runs.
    """
    system = SystemMessage(
        content=SUMMARY_GENERATION_AGENT_PROMPT.format(
            job_description=state.job_description.model_dump_json(),
            skill_tree=state.skill_tree.model_dump_json(),
            candidate_profile=state.candidate_profile.model_dump_json(),
        )
    )
    human = HumanMessage(content="Begin the summary generation per the instructions and input payload.")
    return [system, human]


async def invoke_llm_with_retry(messages: Sequence[Msg], request_id: str) -> GeneratedSummarySchema:
    """
    Invoke the structured-output LLM with timeout and exponential-backoff retries.
    Raises the last exception if all attempts fail.
    """
    last_error: Optional[Exception] = None

    for attempt in range(CONFIG.llm_retries + 1):
        try:
            LOGGER.info(
                "LLM call start",
                extra={"request_id": request_id, "event": "llm_call_start", "attempt": attempt + 1},
            )

            coro = _llm_client.with_structured_output(
                GeneratedSummarySchema
            ).ainvoke(messages)

            result: GeneratedSummarySchema = await asyncio.wait_for(
                coro, timeout=CONFIG.llm_timeout_seconds
            )

            LOGGER.info(
                "LLM call success",
                extra={"request_id": request_id, "event": "llm_call_success", "attempt": attempt + 1},
            )
            return result

        except asyncio.TimeoutError as exc:
            last_error = exc
            LOGGER.error(
                "LLM call timeout",
                extra={"request_id": request_id, "event": "llm_timeout", "attempt": attempt + 1},
            )
        except Exception as exc:
            last_error = exc
            LOGGER.error(
                "LLM call error",
                extra={"request_id": request_id, "event": "llm_error", "attempt": attempt + 1},
            )

        # Backoff before next retry (if any)
        if attempt < CONFIG.llm_retries:
            sleep_secs = CONFIG.llm_retry_backoff_seconds * (2 ** attempt)
            await asyncio.sleep(sleep_secs)

    assert last_error is not None
    raise last_error


# ==============================
# Agent
# ==============================

class SummaryGenerationAgent:
    """
    Generate a structured interview summary from AgentInternalState fields:
      • job_description
      • skill_tree
      • candidate_profile

    Output is coerced to GeneratedSummarySchema and stored at:
      state.generated_summary
    """

    llm_client = _llm_client  # kept for easy swapping/mocking

    @staticmethod
    async def summary_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Build prompt messages, call the LLM (with retries/timeouts),
        write the result to state.generated_summary, and return state.
        """
        request_id = getattr(state, "request_id", None) or str(uuid.uuid4())
        node = "summary_generator"

        # Validate required inputs
        for field in ("job_description", "skill_tree", "candidate_profile"):
            if getattr(state, field, None) is None:
                LOGGER.error(
                    "Missing required field on state: %s",
                    field,
                    extra={"request_id": request_id, "node": node, "event": "state_validation_error"},
                )
                return state  # or raise if you prefer fail-fast

        messages = build_messages(state)

        try:
            summary = await invoke_llm_with_retry(messages, request_id)
            state.generated_summary = summary

            output_for_log = _render_for_log(summary)
            LOGGER.info(
                f"Summary generation completed | output={output_for_log}",
                extra={"request_id": request_id, "node": node, "event": "summary_generated"},
            )
        except Exception:
            LOGGER.error(
                "Summary generation failed",
                extra={"request_id": request_id, "node": node, "event": "summary_failed"},
            )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Build and compile the LangGraph:
          START -> summary_generator
        """
        g = StateGraph(state_schema=AgentInternalState)
        g.add_node("summary_generator", SummaryGenerationAgent.summary_generator)
        g.add_edge(START, "summary_generator")
        return g.compile(checkpointer=checkpointer, name="Summary Generation Agent")


# ==============================
# CLI helper
# ==============================

def draw_graph_ascii() -> None:
    """Print an ASCII preview of the compiled graph (for quick inspection)."""
    try:
        graph = SummaryGenerationAgent.get_graph()
        print(graph.get_graph().draw_ascii())
    except Exception:
        LOGGER.exception("Unable to draw ASCII graph.")


if __name__ == "__main__":
    draw_graph_ascii()
