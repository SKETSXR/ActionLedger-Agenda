import json
import copy
import re
import os
import sys
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import List, Any, Dict, Optional, Tuple, Sequence, Callable, Coroutine

from logging.handlers import TimedRotatingFileHandler
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError, PrivateAttr
from langchain_core.exceptions import OutputParserException
from string import Template

from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import QASetsSchema
from ..prompt.qa_agent_prompt import QA_BLOCK_AGENT_PROMPT
from ..model_handling import llm_qa


# ==============================
# Config (env-overridable)
# ==============================

AGENT_NAME = os.getenv("QA_AGENT_NAME", "qa_block_generation_agent")

LOG_DIR = os.getenv("QA_AGENT_LOG_DIR", "logs")
LOG_LEVEL = getattr(logging, os.getenv("QA_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
LOG_FILE = os.getenv("QA_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
LOG_ROTATE_WHEN = os.getenv("QA_AGENT_LOG_ROTATE_WHEN", "midnight")
LOG_ROTATE_INTERVAL = int(os.getenv("QA_AGENT_LOG_ROTATE_INTERVAL", "1"))
LOG_BACKUP_COUNT = int(os.getenv("QA_AGENT_LOG_BACKUP_COUNT", "365"))

# Retry/timeout knobs (namespaced for this agent)
LLM_TIMEOUT_SECONDS: float = float(os.getenv("QA_AGENT_LLM_TIMEOUT_SECONDS", "90"))
LLM_RETRIES: int = int(os.getenv("QA_AGENT_LLM_RETRIES", "2"))
LLM_BACKOFF_SECONDS: float = float(os.getenv("QA_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

TOOL_TIMEOUT_SECONDS: float = float(os.getenv("QA_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
TOOL_RETRIES: int = int(os.getenv("QA_AGENT_TOOL_RETRIES", "2"))
TOOL_BACKOFF_SECONDS: float = float(os.getenv("QA_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("QA_AGENT_TOOL_MAX_WORKERS", "8")))

# Global retry counter used only for logging iteration counts.
count = 1
SHOW_FULL_TEXT = os.getenv("QA_LOG_SHOW_FULL_TEXT", "0") == "1"
SHOW_FULL_FIELDS = {
    k.strip().lower()
    for k in os.getenv("QA_LOG_SHOW_FULL_FIELDS", "").split(",")
    if k.strip()
}


# ==============================
# Human-style logging
# ==============================

def _build_logger(
    name: str,
    log_dir: str,
    level: int,
    filename: str,
    when: str,
    interval: int,
    backup_count: int,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, filename)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    fh = TimedRotatingFileHandler(
        file_path, when=when, interval=interval, backupCount=backup_count, encoding="utf-8", utc=False
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


LOGGER = _build_logger(
    name=AGENT_NAME,
    log_dir=LOG_DIR,
    level=LOG_LEVEL,
    filename=LOG_FILE,
    when=LOG_ROTATE_WHEN,
    interval=LOG_ROTATE_INTERVAL,
    backup_count=LOG_BACKUP_COUNT,
)


def log_info(msg: str) -> None:
    LOGGER.info(msg)


def log_warning(msg: str) -> None:
    LOGGER.warning(msg)


def log_error(msg: str) -> None:
    LOGGER.error(msg)


# ---------- config ----------
RAW_TEXT_FIELDS = {
    "question_guidelines", "guidelines", "template", "prompt",
    "policy", "notes", "rubric", "examples", "description_md",
}


# ---------- tiny utils ----------
def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def _jsonish(x: Any) -> Any:
    if isinstance(x, str) and _looks_like_json(x):
        try:
            return json.loads(x)
        except Exception:
            return x
    if isinstance(x, dict):
        return {k: _jsonish(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonish(v) for v in x]
    return x


def _compact(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, indent=2) if isinstance(x, (dict, list)) else str(x)
    except Exception:
        return str(x)


def _walk(o: Any, path: Tuple[Any, ...] = ()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, path + (k,))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from _walk(v, path + (i,))
    else:
        yield path, o


# ---------- redaction for compact logs ----------
def _redact(o: Any, *, omit_fields: bool, preview_len: int = 140) -> Any:
    o = _jsonish(o)
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            key = k.lower() if isinstance(k, str) else k

            if isinstance(k, str) and key in RAW_TEXT_FIELDS and isinstance(v, str):
                if omit_fields:
                    continue
                if SHOW_FULL_TEXT or (SHOW_FULL_FIELDS and key in SHOW_FULL_FIELDS):
                    out[k] = v
                else:
                    head = (v.strip().splitlines() or [""])[0]
                    if len(head) > preview_len:
                        head = head[:preview_len].rstrip() + "…"
                    out[k + "_preview"] = head
                    out[k + "_len"] = len(v)
            else:
                out[k] = _redact(v, omit_fields=omit_fields, preview_len=preview_len)
        return out

    if isinstance(o, (list, tuple)):
        return [_redact(v, omit_fields=omit_fields, preview_len=preview_len) for v in o]

    return o


# ---------- main entry ----------
def log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    if not messages:
        return

    # Planned tool calls
    tcalls = getattr(ai_msg, "tool_calls", None)
    if tcalls:
        log_info("Tool plan:")
        for tc in tcalls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            LOGGER.info(f"  planned -> {name} args={_compact(_redact(_jsonish(args), omit_fields=False))}")

    # Trailing tool results
    tool_msgs = []
    i = len(messages) - 1
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        tool_msgs.append(messages[i])
        i -= 1
    if not tool_msgs:
        return

    log_info("Tool results:")
    for tm in tool_msgs:
        content = getattr(tm, "content", None)
        compact = _redact(_jsonish(content), omit_fields=False)
        LOGGER.info(f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_compact(compact)}")


def log_retry_iteration(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    suffix = f" | extra={extra}" if extra else ""
    log_warning(f"Retry {iteration}: {reason}{suffix}")


# ======================================
# Retry / timeout helper for async ops
# ======================================

async def _retry_async(
    op_factory: Callable[[], Coroutine[Any, Any, Any]],
    *,
    retries: int,
    timeout_s: float,
    backoff_base_s: float,
    retry_reason: str,
    iteration_start: int = 1,
) -> Any:
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= retries:
        try:
            return await asyncio.wait_for(op_factory(), timeout=timeout_s)
        except Exception as exc:
            last_exc = exc
            log_retry_iteration(retry_reason, iteration_start + attempt, {"error": str(exc)})
            attempt += 1
            if attempt > retries:
                break
            await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


# ======================================================
# Tool wrapper compatible with bind_tools & ToolNode
# ======================================================

class RetryTool(BaseTool):
    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff: float = PrivateAttr()

    def __init__(self, inner: BaseTool, *, retries: int, timeout_s: float, backoff_base_s: float) -> None:
        name = getattr(inner, "name", inner.__class__.__name__)
        description = getattr(inner, "description", "") or "Retried tool wrapper"
        args_schema = getattr(inner, "args_schema", None)
        super().__init__(name=name, description=description, args_schema=args_schema)
        self._inner = inner
        self._retries = retries
        self._timeout_s = timeout_s
        self._backoff = backoff_base_s

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)

        def _call_once():
            return self._inner._run(*args, **{**kwargs, "config": config})

        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt <= self._retries:
            future = _EXECUTOR.submit(_call_once)
            try:
                return future.result(timeout=self._timeout_s)
            except FuturesTimeout as exc:
                last_exc = exc
                log_retry_iteration(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                last_exc = exc
                log_retry_iteration(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)

        async def _call_once():
            if hasattr(self._inner, "_arun"):
                return await getattr(self._inner, "_arun")(*args, **{**kwargs, "config": config})
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._inner._run(*args, **{**kwargs, "config": config}))

        return await _retry_async(
            _call_once,
            retries=self._retries,
            timeout_s=self._timeout_s,
            backoff_base_s=self._backoff,
            retry_reason=f"tool_async:{self.name}",
        )


# ---------- Inner ReAct state for per-topic QA generation ----------
class _QAInnerState(MessagesState):
    final_response: QASetsSchema


class QABlockGenerationAgent:
    llm_qa = llm_qa

    _RAW_MONGO_TOOLS: List[BaseTool] = get_mongo_tools(llm=llm_qa)
    MONGO_TOOLS: List[BaseTool] = [
        RetryTool(t, retries=TOOL_RETRIES, timeout_s=TOOL_TIMEOUT_SECONDS, backoff_base_s=TOOL_BACKOFF_SECONDS)
        for t in _RAW_MONGO_TOOLS
    ]

    _AGENT_MODEL = llm_qa.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_qa.with_structured_output(QASetsSchema, method="function_calling")

    _compiled_inner_graph = None  # lazy cache

    # ---------- LLM helpers with retries ----------
    @staticmethod
    async def _invoke_agent(messages: Sequence[Any]) -> Any:
        async def _call():
            if hasattr(QABlockGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await QABlockGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, QABlockGenerationAgent._AGENT_MODEL.invoke, messages)

        log_info("Calling LLM (agent)")
        result = await _retry_async(
            _call,
            retries=LLM_RETRIES,
            timeout_s=LLM_TIMEOUT_SECONDS,
            backoff_base_s=LLM_BACKOFF_SECONDS,
            retry_reason="llm:agent",
        )
        log_info("LLM (agent) call succeeded")
        return result

    @staticmethod
    async def _invoke_structured(ai_content: str) -> QASetsSchema:
        payload = [HumanMessage(content=ai_content)]

        async def _call():
            if hasattr(QABlockGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await QABlockGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, QABlockGenerationAgent._STRUCTURED_MODEL.invoke, payload)

        log_info("Calling LLM (structured)")
        result = await _retry_async(
            _call,
            retries=LLM_RETRIES,
            timeout_s=LLM_TIMEOUT_SECONDS,
            backoff_base_s=LLM_BACKOFF_SECONDS,
            retry_reason="llm:structured",
        )
        log_info("LLM (structured) call succeeded")
        return result

    # ---------- Async node impls ----------
    @staticmethod
    async def _agent_node(state: _QAInnerState):
        log_tool_activity(messages=state["messages"], ai_msg=None)
        ai = await QABlockGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _QAInnerState):
        msgs = state["messages"]

        ai_content = None
        for m in reversed(msgs):
            if getattr(m, "type", None) in ("ai", "assistant"):
                if not getattr(m, "tool_calls", None):
                    ai_content = m.content
                    break

        if ai_content is None:
            for m in reversed(msgs):
                if getattr(m, "type", None) in ("ai", "assistant"):
                    ai_content = m.content
                    break

        if ai_content is None:
            ai_content = msgs[-1].content

        final_obj = await QABlockGenerationAgent._invoke_structured(ai_content)
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _QAInnerState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(messages=state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ---------- RunnableLambda wrappers & compile ----------
    @classmethod
    def _get_inner_graph(cls):
        if cls._compiled_inner_graph is not None:
            return cls._compiled_inner_graph

        workflow = StateGraph(_QAInnerState)
        workflow.add_node("agent", RunnableLambda(cls._agent_node))     
        workflow.add_node("respond", RunnableLambda(cls._respond_node)) 
        workflow.add_node("tools", ToolNode(cls.MONGO_TOOLS, tags=["mongo-tools"]))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", cls._should_continue, {"continue": "tools", "respond": "respond"})
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)
        cls._compiled_inner_graph = workflow.compile()
        return cls._compiled_inner_graph

    # ----------------- utilities -----------------
    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "dict"):
            return x.dict()
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _get_topic_name(obj: Any) -> str:
        d = QABlockGenerationAgent._as_dict(obj)
        for k in ("topic", "name", "title", "label"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    @staticmethod
    def _can(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return s

    @staticmethod
    def _extract_topic_name_from_summary(summary_obj: Any) -> str:
        d = QABlockGenerationAgent._as_dict(summary_obj)
        for k in ("topic", "name", "title", "label", "discussion_topic", "heading"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    @staticmethod
    async def _gen_for_topic(
        topic_name: str,
        discussion_summary_json: str,
        deep_dive_nodes_json: str,
        thread_id: str,
        qa_error: str = "",
    ) -> Tuple[Dict[str, Any], str]:
        class AtTemplate(Template):
            delimiter = "@"

        deep_dive_nodes = json.loads(deep_dive_nodes_json or "[]")
        n_blocks = len(deep_dive_nodes)

        COUNT_DIRECTIVE = f"""
        IMPORTANT COUNT RULE:
        - This topic has {n_blocks} deep-dive nodes. Output exactly {n_blocks} QA blocks (one per deep-dive node, in order).
        """

        tpl = AtTemplate(QA_BLOCK_AGENT_PROMPT + COUNT_DIRECTIVE)
        sys_content = tpl.substitute(
            discussion_summary=discussion_summary_json,
            deep_dive_nodes=deep_dive_nodes_json,
            thread_id=thread_id,
            qa_error=qa_error or "",
        )

        sys_message = SystemMessage(content=sys_content)
        trigger_message = HumanMessage(content="Based on the provided instructions please start the process")

        try:
            graph = QABlockGenerationAgent._get_inner_graph()
            result = await graph.ainvoke({"messages": [sys_message, trigger_message]})
            schema = result["final_response"]  # QASetsSchema

            obj = schema.model_dump() if hasattr(schema, "model_dump") else schema
            sets = obj.get("qa_sets", []) or []
            if not sets:
                return {"topic": topic_name, "qa_blocks": []}, "No qa_sets produced."

            one = sets[0]
            one["topic"] = topic_name
            blocks = one.get("qa_blocks", []) or []

            errs = []
            if len(blocks) != n_blocks:
                errs.append(f"Expected {n_blocks} blocks, got {len(blocks)}.")
            for i, b in enumerate(blocks, start=1):
                qi = b.get("qa_items", []) or []
                if len(qi) != 7:
                    errs.append(f"Block {i} must have 7 qa_items, got {len(qi)}.")
                for item in qi:
                    if (item.get("q_type") == "Counter Question" and item.get("q_difficulty") == "Easy"):
                        errs.append(f"Block {i} has an Easy counter (qa_id={item.get('qa_id')}); not allowed.")

            if errs:
                return one, " ; ".join(errs)

            return one, ""

        except (ValidationError, OutputParserException) as e:
            return {"topic": topic_name, "qa_blocks": []}, f"Parser/Schema error: {e}"
        except Exception as e:
            return {"topic": topic_name, "qa_blocks": []}, f"Generation error: {e}"

    @staticmethod
    async def qablock_generator(state: AgentInternalState) -> AgentInternalState:
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")
        if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
            raise ValueError("nodes (topics_with_nodes) are required before QA block generation.")

        raw = state.discussion_summary_per_topic
        if hasattr(raw, "discussion_topics"):
            summaries_list = list(raw.discussion_topics)
        elif isinstance(raw, (list, tuple)):
            summaries_list = list(raw)
        elif isinstance(raw, dict) and "discussion_topics" in raw:
            summaries_list = list(raw["discussion_topics"])
        else:
            raise ValueError("discussion_summary_per_topic has no 'discussion_topics' field")

        final_sets: List[Dict[str, Any]] = []
        accumulated_errs: List[str] = []

        summaries_by_can: Dict[str, Any] = {}
        for s in summaries_list:
            nm = QABlockGenerationAgent._extract_topic_name_from_summary(s)
            summaries_by_can[QABlockGenerationAgent._can(nm)] = s

        DEEP_DIVE_ALIASES = {"deep dive", "deep_dive", "deep-dive", "probe", "follow up", "follow-up"}
        covered = set()

        log_info("QA block generation started")

        for topic_entry in state.nodes.topics_with_nodes:
            topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else dict(topic_entry)
            topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry) or "Unknown"
            ckey = QABlockGenerationAgent._can(topic_name)

            summary_obj = summaries_by_can.get(ckey)
            if summary_obj is None:
                accumulated_errs.append(f"[QABlocks] No summary for '{topic_name}'; skipping topic this round.")
                covered.add(ckey)
                continue

            deep_dive_nodes: List[dict] = []
            for node in (topic_dict.get("nodes") or []):
                qtype = str(node.get("question_type", "")).strip().lower()
                if qtype in DEEP_DIVE_ALIASES:
                    deep_dive_nodes.append(node)

            if not deep_dive_nodes:
                accumulated_errs.append(f"[QABlocks] Topic '{topic_name}' has no deep-dive nodes; skipping this round.")
                covered.add(ckey)
                continue

            summary_json = json.dumps(summary_obj.model_dump() if hasattr(summary_obj, "model_dump") else summary_obj)
            deep_dive_nodes_json = json.dumps(copy.deepcopy(deep_dive_nodes))

            one_set, err = await QABlockGenerationAgent._gen_for_topic(
                topic_name=topic_name,
                discussion_summary_json=summary_json,
                deep_dive_nodes_json=deep_dive_nodes_json,
                thread_id=state.id,
                qa_error=getattr(state, "qa_error", "") or "",
            )

            if err:
                accumulated_errs.append(f"[{topic_name}] {err}")
            blocks = one_set.get("qa_blocks") or []
            if blocks:
                final_sets.append(one_set)
            else:
                accumulated_errs.append(f"[{topic_name}] model returned 0 QA blocks; will retry.")
            covered.add(ckey)

        if final_sets:
            state.qa_blocks = QASetsSchema(qa_sets=final_sets)
            log_info("QA block generation completed")
        else:
            state.qa_blocks = None
            log_warning("QA block generation produced no valid blocks this pass")
            if not accumulated_errs:
                accumulated_errs.append("[QABlocks] No topics produced QA blocks this attempt.")

        if accumulated_errs:
            prev = getattr(state, "qa_error", "") or ""
            state.qa_error = (prev + ("\n" if prev else "") + "\n".join(accumulated_errs)).strip()

        return state

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        global count

        if getattr(state, "qa_blocks", None) is None:
            log_retry_iteration("qa_blocks is None (no valid blocks yet); retrying", count)
            count += 1
            return True

        try:
            QASetsSchema.model_validate(
                state.qa_blocks.model_dump() if hasattr(state.qa_blocks, "model_dump") else state.qa_blocks
            )
        except ValidationError as ve:
            state.qa_error = (
                (getattr(state, "qa_error", "") or "")
                + ("\n" if getattr(state, "qa_error", "") else "")
                + "The previous generated o/p did not follow the given schema as it got following errors:\n"
                "[QABlockGen ValidationError]\n "
                f"{ve}"
            )
            log_retry_iteration("Schema validation failed", count, {"error": str(ve)})
            count += 1
            return True

        try:
            sets = state.qa_blocks.qa_sets if hasattr(state.qa_blocks, "qa_sets") else state.qa_blocks.get("qa_sets", [])
            if any(not (qs.get("qa_blocks") if isinstance(qs, dict) else qs.qa_blocks) for qs in sets):
                log_retry_iteration("At least one topic has 0 qa_blocks after validation; retrying", count)
                count += 1
                return True
        except Exception as e:
            log_retry_iteration("Introspection failed while checking qa_sets; allowing retry", count, {"error": str(e)})
            count += 1
            return True

        return False

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Graph for QA block generation:
        START -> qablock_generator -> (should_regenerate ? qablock_generator : END)
        """
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
        gb.add_edge(START, "qablock_generator")
        gb.add_conditional_edges(
            "qablock_generator",
            QABlockGenerationAgent.should_regenerate,
            {True: "qablock_generator", False: END},
        )
        return gb.compile(checkpointer=checkpointer, name="QA Block Generation Agent")


if __name__ == "__main__":
    graph = QABlockGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
