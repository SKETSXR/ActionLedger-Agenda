# # =============================================================================
# # Module: discussion_summary_per_topic_generation_agent
# # =============================================================================
# # Purpose
# #   Generate a DiscussionSummaryPerTopic for all interview topics by running a
# #   compact ReAct-style inner loop (LLM + Mongo-backed tools) per topic. The
# #   final assistant message is coerced to the typed schema
# #   DiscussionSummaryPerTopicSchema.DiscussionTopic, with exact topic names
# #   enforced to match inputs.
# #
# # Responsibilities
# #   • Build a System prompt per topic from the generated summary + the topic.
# #   • Invoke an LLM bound to Mongo tools (ToolNode) to produce a DiscussionTopic.
# #   • Coerce the final assistant message to the typed schema.
# #   • Run all topics concurrently; collect successful results.
# #   • Validate output-topic set equals input-topic set; route regeneration if not.
# #
# # Data Flow
# #   Inner per-topic loop:
# #     agent (LLM w/ tools) ─► (tools)* ─► respond (coerce to schema)
# #
# #   Outer graph:
# #     START -> discussion_summary_per_topic_generator
# #           -> (should_regenerate ? discussion_summary_per_topic_generator : END)
# #
# # Reliability & Observability
# #   • Timeouts + exponential-backoff retries for LLM and tools.
# #   • Rotating file logs + readable console logs.
# #   • Optional payload logging for tools and final result: off | summary | full.
# #
# # Configuration (Environment Variables)
# #   DISCUSSION_SUMMARY_AGENT_LOG_DIR, DISCUSSION_SUMMARY_AGENT_LOG_FILE, DISCUSSION_SUMMARY_AGENT_LOG_LEVEL
# #   DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_WHEN, DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_INTERVAL, DISCUSSION_SUMMARY_AGENT_LOG_BACKUP_COUNT
# #   DISC_AGENT_LLM_TIMEOUT_SECONDS, DISC_AGENT_LLM_RETRIES, DISC_AGENT_LLM_RETRY_BACKOFF_SECONDS
# #   DISC_AGENT_TOOL_TIMEOUT_SECONDS, DISC_AGENT_TOOL_RETRIES, DISC_AGENT_TOOL_RETRY_BACKOFF_SECONDS
# #   DISC_AGENT_TOOL_MAX_WORKERS
# #   DISC_AGENT_TOOL_LOG_PAYLOAD            off | summary | full
# #   DISC_AGENT_RESULT_LOG_PAYLOAD          off | summary | full
# # =============================================================================


# import asyncio
# import json
# import logging
# import os
# import sys
# import time
# from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
# from dataclasses import dataclass
# from logging.handlers import TimedRotatingFileHandler
# from string import Template
# from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Union

# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.runnables import RunnableLambda
# from langchain_core.tools import BaseTool
# from langgraph.graph import END, START, MessagesState, StateGraph
# from langgraph.prebuilt import ToolNode
# from pydantic import PrivateAttr

# from src.mongo_tools import get_mongo_tools
# from ..model_handling import llm_dts
# from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
#     DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT,
# )
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import DiscussionSummaryPerTopicSchema


# # ==============================
# # Configuration
# # ==============================

# AGENT_NAME = "discussion_summary_agent"


# @dataclass(frozen=True)
# class DiscAgentConfig:
#     log_dir: str = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_DIR", "logs")
#     log_file: str = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
#     log_level: int = getattr(logging, os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
#     log_rotate_when: str = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_WHEN", "midnight")
#     log_rotate_interval: int = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_INTERVAL", "1"))
#     log_backup_count: int = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_BACKUP_COUNT", "365"))

#     llm_timeout_s: float = float(os.getenv("DISC_AGENT_LLM_TIMEOUT_SECONDS", "90"))
#     llm_retries: int = int(os.getenv("DISC_AGENT_LLM_RETRIES", "2"))
#     llm_backoff_base_s: float = float(os.getenv("DISC_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))

#     tool_timeout_s: float = float(os.getenv("DISC_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
#     tool_retries: int = int(os.getenv("DISC_AGENT_TOOL_RETRIES", "2"))
#     tool_backoff_base_s: float = float(os.getenv("DISC_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))
#     tool_max_workers: int = int(os.getenv("DISC_AGENT_TOOL_MAX_WORKERS", "8"))

#     tool_log_payload: str = os.getenv("DISC_AGENT_TOOL_LOG_PAYLOAD", "off").strip().lower()
#     result_log_payload: str = os.getenv("DISC_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()


# CFG = DiscAgentConfig()

# # Shared executor for sync tool calls with timeouts
# _EXECUTOR = ThreadPoolExecutor(max_workers=CFG.tool_max_workers)

# # Global retry counter only for logging iteration counts
# _disc_retry_counter = 1


# # ==============================
# # Logging
# # ==============================

# def _get_logger() -> logging.Logger:
#     logger = logging.getLogger(AGENT_NAME)
#     if logger.handlers:
#         return logger

#     logger.setLevel(CFG.log_level)
#     logger.propagate = False

#     os.makedirs(CFG.log_dir, exist_ok=True)
#     file_path = os.path.join(CFG.log_dir, CFG.log_file)
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

#     console = logging.StreamHandler(sys.stdout)
#     console.setLevel(CFG.log_level)
#     console.setFormatter(fmt)

#     rotating_file = TimedRotatingFileHandler(
#         file_path,
#         when=CFG.log_rotate_when,
#         interval=CFG.log_rotate_interval,
#         backupCount=CFG.log_backup_count,
#         encoding="utf-8",
#         utc=False,
#         delay=True,
#     )
#     logging.raiseExceptions = False  # never raise on logging I/O in production
#     rotating_file.setLevel(CFG.log_level)
#     rotating_file.setFormatter(fmt)

#     logger.addHandler(console)
#     logger.addHandler(rotating_file)
#     return logger


# LOGGER = _get_logger()


# def _log_info(msg: str) -> None:
#     LOGGER.info(msg)


# def _log_warn(msg: str) -> None:
#     LOGGER.warning(msg)


# # ==============================
# # Small JSON / logging helpers
# # ==============================

# def _looks_like_json(text: str) -> bool:
#     text = text.strip()
#     return (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))


# def _jsonish(value: Any) -> Any:
#     if isinstance(value, str) and _looks_like_json(value):
#         try:
#             return json.loads(value)
#         except Exception:
#             return value
#     if isinstance(value, dict):
#         return {k: _jsonish(v) for k, v in value.items()}
#     if isinstance(value, (list, tuple)):
#         return [_jsonish(v) for v in value]
#     return value


# def _compact(value: Any) -> str:
#     try:
#         return json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
#     except Exception:
#         return str(value)


# def _pydantic_to_obj(obj: Any) -> Any:
#     # pydantic v2
#     if hasattr(obj, "model_dump_json"):
#         try:
#             return json.loads(obj.model_dump_json())
#         except Exception:
#             pass
#     if hasattr(obj, "model_dump"):
#         try:
#             return obj.model_dump()
#         except Exception:
#             pass
#     # pydantic v1 / dict-like
#     if hasattr(obj, "dict"):
#         try:
#             return obj.dict()
#         except Exception:
#             pass
#     return obj


# def _summarize_tool_payload(payload: Any) -> str:
#     try:
#         obj = _jsonish(payload)
#         if isinstance(obj, dict):
#             keys = list(obj.keys())[:6]
#             return f"keys={keys} ok={obj.get('ok')} count={obj.get('count')} data={'yes' if 'data' in obj else 'no'}"
#         if isinstance(obj, list):
#             return f"list(len={len(obj)})"
#         return type(obj).__name__
#     except Exception:
#         return "<unavailable>"


# def _render_tool_payload(payload: Any) -> str:
#     mode = CFG.tool_log_payload
#     if mode == "off":
#         return "<hidden>"
#     if mode == "summary":
#         return _summarize_tool_payload(payload)
#     return _compact(_jsonish(payload))


# def _summarize_final_result(payload: Any) -> str:
#     """
#     Summarize DiscussionSummaryPerTopicSchema:
#       - count of topics
#       - first few topic names
#     """
#     try:
#         data = _pydantic_to_obj(payload)
#         topics = None

#         if isinstance(data, dict) and "discussion_topics" in data:
#             topics = data["discussion_topics"]
#         elif isinstance(payload, DiscussionSummaryPerTopicSchema):
#             topics = payload.discussion_topics

#         names: List[str] = []
#         if isinstance(topics, list):
#             for t in topics[:8]:
#                 nm = None
#                 if isinstance(t, dict):
#                     nm = t.get("topic") or t.get("name") or t.get("title") or t.get("label")
#                 else:
#                     nm = getattr(t, "topic", None) or getattr(t, "name", None) or getattr(t, "title", None)
#                 if isinstance(nm, str) and nm.strip():
#                     names.append(nm.strip())
#             suffix = "..." if len(topics) > 8 else ""
#             return f"discussion_topics.len={len(topics)} names={names}{suffix}"

#         if isinstance(data, dict):
#             return f"keys={list(data.keys())[:8]}"
#         if isinstance(data, list):
#             return f"list(len={len(data)})"
#         return type(data).__name__
#     except Exception:
#         return "<unavailable>"


# def _render_final_result(payload: Any) -> str:
#     mode = CFG.result_log_payload
#     if mode == "off":
#         return "<hidden>"
#     if mode == "summary":
#         return _summarize_final_result(payload)
#     return _compact(_pydantic_to_obj(payload))


# def _log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
#     if not messages:
#         return

#     # Planned tool calls
#     planned = getattr(ai_msg, "tool_calls", None)
#     if planned:
#         _log_info("Tool plan:")
#         for tc in planned:
#             name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
#             args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
#             LOGGER.info(f"  planned -> {name} args={_render_tool_payload(args)}")

#     # Trailing tool results in buffer
#     tool_msgs: List[Any] = []
#     i = len(messages) - 1
#     while i >= 0 and getattr(messages[i], "type", None) == "tool":
#         tool_msgs.append(messages[i])
#         i -= 1
#     if not tool_msgs:
#         return

#     _log_info("Tool results:")
#     for tm in tool_msgs:
#         content = getattr(tm, "content", None)
#         LOGGER.info(f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_render_tool_payload(content)}")


# def _log_retry(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
#     suffix = f" | extra={extra}" if extra else ""
#     _log_warn(f"Retry {iteration}: {reason}{suffix}")


# # ==============================
# # Async retry helper
# # ==============================

# async def _retry_async(
#     op_factory: Callable[[], Coroutine[Any, Any, Any]],
#     *,
#     retries: int,
#     timeout_s: float,
#     backoff_base_s: float,
#     retry_reason: str,
#     iteration_start: int = 1,
# ) -> Any:
#     """Run op_factory with timeout and exponential-backoff retries."""
#     attempt = 0
#     last_exc: Optional[BaseException] = None
#     while attempt <= retries:
#         try:
#             return await asyncio.wait_for(op_factory(), timeout=timeout_s)
#         except Exception as exc:
#             last_exc = exc
#             _log_retry(retry_reason, iteration_start + attempt, {"error": str(exc)})
#             attempt += 1
#             if attempt > retries:
#                 break
#             await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))
#     assert last_exc is not None
#     raise last_exc


# # ==============================
# # Tool wrapper (timeout + retry)
# # ==============================

# class RetryTool(BaseTool):
#     """Wrap BaseTool with timeout + retries for both sync and async paths."""

#     _inner: BaseTool = PrivateAttr()
#     _retries: int = PrivateAttr()
#     _timeout_s: float = PrivateAttr()
#     _backoff_base_s: float = PrivateAttr()

#     def __init__(self, inner: BaseTool, *, retries: int, timeout_s: float, backoff_base_s: float) -> None:
#         name = getattr(inner, "name", inner.__class__.__name__)
#         description = getattr(inner, "description", "") or "Retried tool wrapper"
#         args_schema = getattr(inner, "args_schema", None)
#         super().__init__(name=name, description=description, args_schema=args_schema)

#         self._inner = inner
#         self._retries = retries
#         self._timeout_s = timeout_s
#         self._backoff_base_s = backoff_base_s

#     def _run(self, *args, **kwargs):
#         """Sync path via threadpool with timeout + retries."""
#         config = kwargs.pop("config", None)

#         def _call_once():
#             return self._inner._run(*args, **{**kwargs, "config": config})

#         attempt = 0
#         last_exc: Optional[BaseException] = None
#         while attempt <= self._retries:
#             fut = _EXECUTOR.submit(_call_once)
#             try:
#                 return fut.result(timeout=self._timeout_s)
#             except FuturesTimeout as exc:
#                 last_exc = exc
#                 _log_retry(f"tool_timeout:{self.name}", attempt + 1)
#             except BaseException as exc:
#                 last_exc = exc
#                 _log_retry(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
#             attempt += 1
#             if attempt <= self._retries:
#                 time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
#         assert last_exc is not None
#         raise last_exc

#     async def _arun(self, *args, **kwargs):
#         """Async path with timeout + retries; falls back to sync via executor."""
#         config = kwargs.pop("config", None)

#         async def _call_once():
#             if hasattr(self._inner, "_arun"):
#                 return await self._inner._arun(*args, **{**kwargs, "config": config})
#             loop = asyncio.get_running_loop()
#             return await loop.run_in_executor(None, lambda: self._inner._run(*args, **{**kwargs, "config": config}))

#         return await _retry_async(
#             _call_once,
#             retries=self._retries,
#             timeout_s=self._timeout_s,
#             backoff_base_s=self._backoff_base_s,
#             retry_reason=f"tool_async:{self.name}",
#         )


# # ==============================
# # Inner per-topic ReAct loop
# # ==============================

# class _PerTopicState(MessagesState):
#     """State container for the inner Mongo-enabled loop per topic."""
#     final_response: DiscussionSummaryPerTopicSchema.DiscussionTopic


# Msg = Union[SystemMessage, HumanMessage]


# class PerTopicDiscussionSummaryGenerationAgent:
#     """Generates a DiscussionTopic for a single topic via an inner ReAct loop."""

#     llm = llm_dts

#     # Tools (wrapped with retry/timeout)
#     _RAW_TOOLS = get_mongo_tools(llm=llm)
#     TOOLS = [
#         RetryTool(t, retries=CFG.tool_retries, timeout_s=CFG.tool_timeout_s, backoff_base_s=CFG.tool_backoff_base_s)
#         for t in _RAW_TOOLS
#     ]

#     _AGENT_MODEL = llm.bind_tools(TOOLS)
#     _STRUCTURED_MODEL = llm.with_structured_output(
#         DiscussionSummaryPerTopicSchema.DiscussionTopic
#     )

#     _compiled_graph = None  # cache

#     # ----- LLM invokers -----

#     @staticmethod
#     async def _invoke_agent(messages: Sequence[Msg]) -> Any:
#         async def _call():
#             if hasattr(PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL, "ainvoke"):
#                 return await PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.ainvoke(messages)
#             loop = asyncio.get_running_loop()
#             return await loop.run_in_executor(
#                 None, PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.invoke, messages
#             )

#         _log_info("Calling LLM (agent)")
#         ai = await _retry_async(
#             _call,
#             retries=CFG.llm_retries,
#             timeout_s=CFG.llm_timeout_s,
#             backoff_base_s=CFG.llm_backoff_base_s,
#             retry_reason="llm:agent",
#         )
#         _log_info("LLM (agent) call succeeded")
#         return ai

#     @staticmethod
#     async def _invoke_structured(ai_content: str) -> DiscussionSummaryPerTopicSchema.DiscussionTopic:
#         payload = [HumanMessage(content=ai_content)]

#         async def _call():
#             if hasattr(PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
#                 return await PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
#             loop = asyncio.get_running_loop()
#             return await loop.run_in_executor(
#                 None, PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.invoke, payload
#             )

#         _log_info("Calling LLM (structured)")
#         obj = await _retry_async(
#             _call,
#             retries=CFG.llm_retries,
#             timeout_s=CFG.llm_timeout_s,
#             backoff_base_s=CFG.llm_backoff_base_s,
#             retry_reason="llm:structured",
#         )
#         _log_info("LLM (structured) call succeeded")
#         return obj

#     # ----- Inner graph nodes (async) -----

#     @staticmethod
#     async def _agent_node(state: _PerTopicState):
#         _log_tool_activity(state["messages"], ai_msg=None)
#         ai = await PerTopicDiscussionSummaryGenerationAgent._invoke_agent(state["messages"])
#         return {"messages": [ai]}

#     @staticmethod
#     async def _respond_node(state: _PerTopicState):
#         msgs = state["messages"]
#         ai_content: Optional[str] = None

#         # Prefer last assistant msg without tool calls
#         for m in reversed(msgs):
#             if getattr(m, "type", None) in ("ai", "assistant") and not getattr(m, "tool_calls", None):
#                 ai_content = m.content
#                 break
#         # Fallback: any assistant msg
#         if ai_content is None:
#             for m in reversed(msgs):
#                 if getattr(m, "type", None) in ("ai", "assistant"):
#                     ai_content = m.content
#                     break
#         # Final fallback: last message content
#         if ai_content is None:
#             ai_content = msgs[-1].content

#         final_obj = await PerTopicDiscussionSummaryGenerationAgent._invoke_structured(ai_content)
#         return {"final_response": final_obj}

#     @staticmethod
#     def _should_continue(state: _PerTopicState):
#         last = state["messages"][-1]
#         if getattr(last, "tool_calls", None):
#             _log_tool_activity(state["messages"], ai_msg=last)
#             return "continue"
#         return "respond"

#     # ----- RunnableLambda wrappers (avoid staticmethod binding ambiguity) -----

#     @staticmethod
#     async def _agent_node_async(state: _PerTopicState):
#         return await PerTopicDiscussionSummaryGenerationAgent._agent_node(state)

#     @staticmethod
#     async def _respond_node_async(state: _PerTopicState):
#         return await PerTopicDiscussionSummaryGenerationAgent._respond_node(state)

#     # ----- Compile inner ReAct graph -----

#     @classmethod
#     def _get_graph(cls):
#         if cls._compiled_graph is not None:
#             return cls._compiled_graph

#         g = StateGraph(_PerTopicState)
#         g.add_node("agent", RunnableLambda(cls._agent_node_async))
#         g.add_node("respond", RunnableLambda(cls._respond_node_async))
#         g.add_node("tools", ToolNode(cls.TOOLS, tags=["mongo-tools"]))
#         g.set_entry_point("agent")
#         g.add_conditional_edges("agent", cls._should_continue, {"continue": "tools", "respond": "respond"})
#         g.add_edge("tools", "agent")
#         g.add_edge("respond", END)

#         cls._compiled_graph = g.compile()
#         return cls._compiled_graph

#     @staticmethod
#     async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any], thread_id: str):
#         class AtTemplate(Template):
#             delimiter = "@"

#         tpl = AtTemplate(DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT)
#         sys_content = tpl.substitute(
#             generated_summary=generated_summary_json,
#             interview_topic=json.dumps(topic, ensure_ascii=False),
#             thread_id=thread_id,
#         )

#         sys_msg = SystemMessage(content=sys_content)
#         trigger = HumanMessage(content="Based on the provided instructions please start the process")

#         graph = PerTopicDiscussionSummaryGenerationAgent._get_graph()
#         result = await graph.ainvoke({"messages": [sys_msg, trigger]})
#         return result["final_response"]


# # ==============================
# # Outer graph: nodes & routing
# # ==============================

# class PerTopicDiscussionSummaryAgent:
#     """Outer flow that runs inner per-topic loops concurrently and enforces policy."""

#     @staticmethod
#     async def should_regenerate(state: AgentInternalState) -> bool:
#         """
#         Regenerate if the set of topics in the output does not exactly match the
#         set of input topics. Includes a guard to avoid infinite loops if nothing
#         was produced (single extra retry).
#         """
#         global _disc_retry_counter

#         # Guard: if nothing produced, allow at most 1 retry under this condition
#         if not getattr(state, "discussion_summary_per_topic", None) or \
#            not getattr(state.discussion_summary_per_topic, "discussion_topics", None):
#             _log_retry("No discussion topics produced; retrying once", _disc_retry_counter)
#             _disc_retry_counter += 1
#             return _disc_retry_counter <= 2  # retry only once for the "no output" case

#         input_topics = {t.topic for t in state.interview_topics.interview_topics}
#         output_topics = {dt.topic for dt in state.discussion_summary_per_topic.discussion_topics}

#         if input_topics != output_topics:
#             missing = input_topics - output_topics
#             extra = output_topics - input_topics
#             _log_retry("Topic mismatch", _disc_retry_counter, {"missing": missing, "extra": extra})
#             _disc_retry_counter += 1
#             return True
#         return False

#     @staticmethod
#     async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
#         """
#         Generate summaries for all topics concurrently:
#           1) Normalize the input topics
#           2) Serialize generated_summary for prompting
#           3) Launch one inner-graph run per topic via asyncio.gather
#           4) Force output topic names to match input names exactly
#         """
#         # Normalize topics list coming from parent state
#         try:
#             topics_list: List[Dict[str, Any]] = [t.model_dump() for t in state.interview_topics.interview_topics]
#         except Exception:
#             topics_list = state.interview_topics  # already a list[dict]

#         if not isinstance(topics_list, list) or len(topics_list) == 0:
#             raise ValueError("interview_topics must be a non-empty list[dict]")

#         # Serialize generated_summary for prompt
#         try:
#             generated_summary_json = state.generated_summary.model_dump_json()
#         except Exception:
#             generated_summary_json = json.dumps(state.generated_summary, ensure_ascii=False)

#         # Run all topic calls concurrently (each via inner graph)
#         tasks = [
#             asyncio.create_task(
#                 PerTopicDiscussionSummaryGenerationAgent._one_topic_call(generated_summary_json, topic, state.id)
#             )
#             for topic in topics_list
#         ]
#         results = await asyncio.gather(*tasks, return_exceptions=True)

#         # Collect successful DiscussionTopic entries and enforce exact topic names
#         discussion_topics: List[Any] = []
#         for idx, result in enumerate(results):
#             if isinstance(result, Exception):
#                 _log_warn(f"Topic {idx} summarization failed: {result}")
#                 continue

#             input_topic_name = (
#                 topics_list[idx].get("topic")
#                 or topics_list[idx].get("name")
#                 or topics_list[idx].get("title")
#                 or "Unknown"
#             )
#             try:
#                 if hasattr(result, "model_copy"):  # pydantic v2
#                     result = result.model_copy(update={"topic": input_topic_name})
#                 elif hasattr(result, "copy"):  # pydantic v1
#                     result = result.copy(update={"topic": input_topic_name})
#                 elif isinstance(result, dict):
#                     result["topic"] = input_topic_name
#                 else:
#                     setattr(result, "topic", input_topic_name)
#                 discussion_topics.append(result)
#             except Exception as exc:
#                 _log_warn(f"Failed to append structured response for topic index {idx}: {exc}")

#         state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(discussion_topics=discussion_topics)

#         # Gated final-output logging
#         rendered = _render_final_result(state.discussion_summary_per_topic.model_dump_json(indent=2))
#         _log_info(f"Per-topic discussion summaries generated | output={rendered}")

#         return state

#     @staticmethod
#     def get_graph(checkpointer=None):
#         """
#         Topic-wise discussion summary graph:
#         START -> discussion_summary_per_topic_generator
#                -> (should_regenerate ? discussion_summary_per_topic_generator : END)
#         """
#         g = StateGraph(AgentInternalState)
#         g.add_node(
#             "discussion_summary_per_topic_generator",
#             PerTopicDiscussionSummaryAgent.discussion_summary_per_topic_generator,
#         )
#         g.add_edge(START, "discussion_summary_per_topic_generator")
#         g.add_conditional_edges(
#             "discussion_summary_per_topic_generator",
#             PerTopicDiscussionSummaryAgent.should_regenerate,
#             {False: END, True: "discussion_summary_per_topic_generator"},
#         )
#         g.add_edge("discussion_summary_per_topic_generator", END)
#         return g.compile(checkpointer=checkpointer, name="PerTopicDiscussionSummaryGenerationAgent")


# if __name__ == "__main__":
#     graph = PerTopicDiscussionSummaryAgent.get_graph()
#     print(graph.get_graph().draw_ascii())


# =============================================================================
# Module: discussion_summary_per_topic_agent
# =============================================================================
# Purpose
#   Generate a structured discussion summary for each interview topic using an
#   inner ReAct loop (Mongo-backed tools) and aggregate the results at the outer
#   graph level. Includes robust JSON coercion, timeouts, retries, and logging.
#
# Graphs
#   Inner (per topic):
#     agent ─► (tools)* ─► respond
#       - agent: tool-enabled LLM plans and may call tools
#       - tools: ToolNode executes planned tool calls
#       - respond: coerce final tool-free assistant content to schema
#
#   Outer:
#     START ──► discussion_summary_per_topic_generator
#               ├─► discussion_summary_per_topic_generator (should_regenerate=True)
#               └─► END                                  (should_regenerate=False)
#
# Environment Variables
#   DISCUSSION_SUMMARY_AGENT_LOG_DIR, DISCUSSION_SUMMARY_AGENT_LOG_FILE,
#   DISCUSSION_SUMMARY_AGENT_LOG_LEVEL, DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_WHEN,
#   DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_INTERVAL, DISCUSSION_SUMMARY_AGENT_LOG_BACKUP_COUNT
#
#   DISC_AGENT_LLM_TIMEOUT_SECONDS, DISC_AGENT_LLM_RETRIES, DISC_AGENT_LLM_RETRY_BACKOFF_SECONDS
#   DISC_AGENT_TOOL_TIMEOUT_SECONDS, DISC_AGENT_TOOL_RETRIES, DISC_AGENT_TOOL_RETRY_BACKOFF_SECONDS
#   DISC_AGENT_TOOL_MAX_WORKERS
#
#   DISC_AGENT_TOOL_LOG_PAYLOAD (off|summary|full)
#   DISC_AGENT_RESULT_LOG_PAYLOAD (off|summary|full)
# =============================================================================

import asyncio
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from logging.handlers import TimedRotatingFileHandler
from string import Template
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, Union

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import PrivateAttr

from src.mongo_tools import get_mongo_tools
from ..model_handling import llm_dts
from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
    DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT,
)
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import DiscussionSummaryPerTopicSchema

# Ensure nested forward refs are resolved (Pydantic v2 safe)
try:
    DiscussionSummaryPerTopicSchema.model_rebuild()
except Exception:
    pass


# ==============================
# Configuration
# ==============================

AGENT_NAME = "discussion_summary_agent"


@dataclass(frozen=True)
class DiscAgentConfig:
    # logging
    log_dir: str = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_DIR", "logs")
    log_file: str = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_FILE", f"{AGENT_NAME}.log")
    log_level: int = getattr(logging, os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_rotate_when: str = os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_WHEN", "midnight")
    log_rotate_interval: int = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_ROTATE_INTERVAL", "1"))
    log_backup_count: int = int(os.getenv("DISCUSSION_SUMMARY_AGENT_LOG_BACKUP_COUNT", "365"))
    # LLM
    llm_timeout_s: float = float(os.getenv("DISC_AGENT_LLM_TIMEOUT_SECONDS", "90"))
    llm_retries: int = int(os.getenv("DISC_AGENT_LLM_RETRIES", "2"))
    llm_backoff_base_s: float = float(os.getenv("DISC_AGENT_LLM_RETRY_BACKOFF_SECONDS", "2.5"))
    # Tools
    tool_timeout_s: float = float(os.getenv("DISC_AGENT_TOOL_TIMEOUT_SECONDS", "30"))
    tool_retries: int = int(os.getenv("DISC_AGENT_TOOL_RETRIES", "2"))
    tool_backoff_base_s: float = float(os.getenv("DISC_AGENT_TOOL_RETRY_BACKOFF_SECONDS", "1.5"))
    tool_max_workers: int = int(os.getenv("DISC_AGENT_TOOL_MAX_WORKERS", "8"))
    # Logging verbosity
    tool_log_payload: str = os.getenv("DISC_AGENT_TOOL_LOG_PAYLOAD", "off").strip().lower()       # off|summary|full
    result_log_payload: str = os.getenv("DISC_AGENT_RESULT_LOG_PAYLOAD", "off").strip().lower()   # off|summary|full


CFG = DiscAgentConfig()

# Shared executor for sync tool calls with timeouts
_EXECUTOR = ThreadPoolExecutor(max_workers=CFG.tool_max_workers)

# Global retry counter (for logging iterations only)
_disc_retry_counter = 1


# ==============================
# Logging
# ==============================

def _get_logger() -> logging.Logger:
    logger = logging.getLogger(AGENT_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(CFG.log_level)
    logger.propagate = False

    os.makedirs(CFG.log_dir, exist_ok=True)
    file_path = os.path.join(CFG.log_dir, CFG.log_file)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(CFG.log_level)
    console.setFormatter(fmt)

    rotating_file = TimedRotatingFileHandler(
        file_path,
        when=CFG.log_rotate_when,
        interval=CFG.log_rotate_interval,
        backupCount=CFG.log_backup_count,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    logging.raiseExceptions = False  # production: never raise on logging I/O
    rotating_file.setLevel(CFG.log_level)
    rotating_file.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(rotating_file)
    return logger


LOGGER = _get_logger()


def _log_info(msg: str) -> None:
    LOGGER.info(msg)


def _log_warn(msg: str) -> None:
    LOGGER.warning(msg)


# ==============================
# Small JSON / logging helpers
# ==============================

def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]"))


def _jsonish(value: Any) -> Any:
    if isinstance(value, str) and _looks_like_json(value):
        try:
            return json.loads(value)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    return value


def _compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
    except Exception:
        return str(value)


def _pydantic_to_obj(obj: Any) -> Any:
    if hasattr(obj, "model_dump_json"):
        try:
            return json.loads(obj.model_dump_json())
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return obj


def _summarize_final_result(payload: Any) -> str:
    """
    Summarize DiscussionSummaryPerTopicSchema:
      - count of topics
      - first few topic names
    """
    try:
        data = _pydantic_to_obj(payload)
        topics = None

        if isinstance(payload, DiscussionSummaryPerTopicSchema):
            topics = payload.discussion_topics
        elif isinstance(data, dict) and "discussion_topics" in data:
            topics = data["discussion_topics"]

        names: List[str] = []
        if isinstance(topics, list):
            for t in topics[:8]:
                nm = None
                if isinstance(t, dict):
                    nm = t.get("topic") or t.get("name") or t.get("title") or t.get("label")
                else:
                    nm = getattr(t, "topic", None) or getattr(t, "name", None) or getattr(t, "title", None)
                if isinstance(nm, str) and nm.strip():
                    names.append(nm.strip())
            suffix = "..." if len(topics) > 8 else ""
            return f"discussion_topics.len={len(topics)} names={names}{suffix}"

        if isinstance(data, dict):
            return f"keys={list(data.keys())[:8]}"
        if isinstance(data, list):
            return f"list(len={len(data)})"
        return type(data).__name__
    except Exception:
        return "<unavailable>"


def _render_tool_payload(payload: Any) -> str:
    mode = CFG.tool_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        try:
            obj = _jsonish(payload)
            if isinstance(obj, dict):
                keys = list(obj.keys())[:6]
                return f"keys={keys} ok={obj.get('ok')} count={obj.get('count')} data={'yes' if 'data' in obj else 'no'}"
            if isinstance(obj, list):
                return f"list(len={len(obj)})"
            return type(obj).__name__
        except Exception:
            return "<unavailable>"
    return _compact(_jsonish(payload))


def _render_final_result(payload: Any) -> str:
    mode = CFG.result_log_payload
    if mode == "off":
        return "<hidden>"
    if mode == "summary":
        return _summarize_final_result(payload)
    return _compact(_pydantic_to_obj(payload))


def _log_tool_activity(messages: Sequence[Any], ai_msg: Optional[Any] = None) -> None:
    if not messages:
        return

    # Planned tool calls
    planned = getattr(ai_msg, "tool_calls", None)
    if planned:
        _log_info("Tool plan:")
        for tc in planned:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            LOGGER.info(f"  planned -> {name} args={_render_tool_payload(args)}")

    # Trailing tool results
    tool_msgs: List[Any] = []
    i = len(messages) - 1
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        tool_msgs.append(messages[i])
        i -= 1
    if not tool_msgs:
        return

    _log_info("Tool results:")
    for tm in tool_msgs:
        content = getattr(tm, "content", None)
        LOGGER.info(f"  result -> id={getattr(tm, 'tool_call_id', None)} data={_render_tool_payload(content)}")


def _log_retry(reason: str, iteration: int, extra: Optional[dict] = None) -> None:
    suffix = f" | extra={extra}" if extra else ""
    _log_warn(f"Retry {iteration}: {reason}{suffix}")


# ==============================
# Coercion helpers (wrapped -> flat)
# ==============================

_STEP_KEYS = ("Opening", "DirectQuestion", "DeepDive")

def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    return [str(x)]


def _normalize_one_step(step: Any) -> Optional[Dict[str, Any]]:
    """
    Accept either:
      • {"type": "...", "description": "...", ...}
      • {"Opening": {...}} / {"DirectQuestion": {...}} / {"DeepDive": {...}}
    Return flat dict with required keys or None if cannot normalize.
    """
    if not isinstance(step, dict):
        return None

    # Flat form already
    if "type" in step and "description" in step and "guidelines" in step:
        tval = step.get("type", "")
        if isinstance(tval, str) and tval.strip().lower().replace(" ", "") == "directquestion":
            tval = "Direct Question"
        return {
            "type": str(tval),
            "description": str(step.get("description", "")),
            "guidelines": str(step.get("guidelines", "")),
            "focus_areas": _as_list_str(step.get("focus_areas")),
            "reference_sources": _as_list_str(step.get("reference_sources")),
        }

    # Wrapped form
    for k in _STEP_KEYS:
        if k in step and isinstance(step[k], dict):
            inner = step[k]
            tval = "Direct Question" if k == "DirectQuestion" else k
            return {
                "type": tval,
                "description": str(inner.get("description", "")),
                "guidelines": str(inner.get("guidelines", "")),
                "focus_areas": _as_list_str(inner.get("focus_areas")),
                "reference_sources": _as_list_str(inner.get("reference_sources")),
            }

    return None


def _normalize_topic_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Accept a dict that may contain a wrapped sequence.
    Return a dict matching DiscussionSummaryPerTopicSchema.DiscussionTopic.
    """
    if not isinstance(obj, dict):
        return None

    topic = str(obj.get("topic") or obj.get("name") or obj.get("title") or "").strip()
    guidelines = str(obj.get("guidelines", ""))
    focus_areas_covered = _as_list_str(obj.get("focus_areas_covered"))
    reference_material = _as_list_str(obj.get("reference_material"))

    seq_in = obj.get("sequence") or []
    if not isinstance(seq_in, list):
        seq_in = []

    seq_out: List[Dict[str, Any]] = []
    for step in seq_in:
        norm = _normalize_one_step(step)
        if norm is not None:
            seq_out.append(norm)

    if not seq_out:
        return None

    return {
        "topic": topic or "Unknown",
        "sequence": seq_out,
        "guidelines": guidelines,
        "focus_areas_covered": focus_areas_covered,
        "reference_material": reference_material,
    }


# ==============================
# JSON extraction helpers
# ==============================

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)

def _find_fenced_json(txt: str) -> Optional[str]:
    m = _CODE_FENCE_RE.search(txt)
    return m.group(1) if m else None


def _find_balanced_json(txt: str) -> Optional[str]:
    """Find the first balanced top-level JSON object or array in arbitrary text."""
    starts = [("{", "}"), ("[", "]")]
    for open_ch, close_ch in starts:
        start_idx = txt.find(open_ch)
        while start_idx != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start_idx, len(txt)):
                ch = txt[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            candidate = txt[start_idx : i + 1]
                            try:
                                json.loads(candidate)
                                return candidate
                            except Exception:
                                break
            start_idx = txt.find(open_ch, start_idx + 1)
    return None


def _extract_json_str(txt: str) -> Optional[str]:
    if not isinstance(txt, str) or not txt.strip():
        return None
    fenced = _find_fenced_json(txt)
    if fenced:
        return fenced
    if _looks_like_json(txt.strip()):
        return txt.strip()
    return _find_balanced_json(txt)


# ==============================
# Async retry helper
# ==============================

async def _retry_async(
    op_factory: Callable[[], Coroutine[Any, Any, Any]],
    *,
    retries: int,
    timeout_s: float,
    backoff_base_s: float,
    retry_reason: str,
    iteration_start: int = 1,
) -> Any:
    """Run op_factory with timeout and exponential-backoff retries."""
    attempt = 0
    last_exc: Optional[BaseException] = None
    while attempt <= retries:
        try:
            return await asyncio.wait_for(op_factory(), timeout=timeout_s)
        except Exception as exc:
            last_exc = exc
            _log_retry(retry_reason, iteration_start + attempt, {"error": str(exc)})
            attempt += 1
            if attempt > retries:
                break
            await asyncio.sleep(backoff_base_s * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


# ==============================
# Tool wrapper (timeout + retry)
# ==============================

class RetryTool(BaseTool):
    """Wrap BaseTool with timeout + retries for both sync and async paths."""

    _inner: BaseTool = PrivateAttr()
    _retries: int = PrivateAttr()
    _timeout_s: float = PrivateAttr()
    _backoff_base_s: float = PrivateAttr()

    def __init__(self, inner: BaseTool, *, retries: int, timeout_s: float, backoff_base_s: float) -> None:
        name = getattr(inner, "name", inner.__class__.__name__)
        description = getattr(inner, "description", "") or "Retried tool wrapper"
        args_schema = getattr(inner, "args_schema", None)
        super().__init__(name=name, description=description, args_schema=args_schema)

        self._inner = inner
        self._retries = retries
        self._timeout_s = timeout_s
        self._backoff_base_s = backoff_base_s

    def _run(self, *args, **kwargs):
        """Sync path via threadpool with timeout + retries."""
        config = kwargs.pop("config", None)

        def _call_once():
            return self._inner._run(*args, **{**kwargs, "config": config})

        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt <= self._retries:
            fut = _EXECUTOR.submit(_call_once)
            try:
                return fut.result(timeout=self._timeout_s)
            except FuturesTimeout as exc:
                last_exc = exc
                _log_retry(f"tool_timeout:{self.name}", attempt + 1)
            except BaseException as exc:
                last_exc = exc
                _log_retry(f"tool_error:{self.name}", attempt + 1, {"error": str(exc)})
            attempt += 1
            if attempt <= self._retries:
                time.sleep(self._backoff_base_s * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    async def _arun(self, *args, **kwargs):
        """Async path with timeout + retries; falls back to sync via executor."""
        config = kwargs.pop("config", None)

        async def _call_once():
            if hasattr(self._inner, "_arun"):
                return await self._inner._arun(*args, **{**kwargs, "config": config})
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._inner._run(*args, **{**kwargs, "config": config}))

        return await _retry_async(
            _call_once,
            retries=self._retries,
            timeout_s=self._timeout_s,
            backoff_base_s=self._backoff_base_s,
            retry_reason=f"tool_async:{self.name}",
        )


# ==============================
# Inner per-topic ReAct loop
# ==============================

class _PerTopicState(MessagesState):
    """State container for the inner Mongo-enabled loop per topic."""
    final_response: DiscussionSummaryPerTopicSchema.DiscussionTopic


Msg = Union[SystemMessage, HumanMessage]


class PerTopicDiscussionSummaryGenerationAgent:
    """Generates ONE DiscussionTopic for a single topic via an inner ReAct loop."""

    llm = llm_dts

    _RAW_TOOLS = get_mongo_tools(llm=llm)
    TOOLS = [
        RetryTool(t, retries=CFG.tool_retries, timeout_s=CFG.tool_timeout_s, backoff_base_s=CFG.tool_backoff_base_s)
        for t in _RAW_TOOLS
    ]

    _AGENT_MODEL = llm.bind_tools(TOOLS)
    _STRUCTURED_MODEL = llm.with_structured_output(DiscussionSummaryPerTopicSchema.DiscussionTopic)

    _compiled_graph = None  # cache

    # ----- LLM invokers -----

    @staticmethod
    async def _invoke_agent(messages: Sequence[Msg]) -> Any:
        async def _call():
            if hasattr(PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL, "ainvoke"):
                return await PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.ainvoke(messages)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, PerTopicDiscussionSummaryGenerationAgent._AGENT_MODEL.invoke, messages
            )

        _log_info("Calling LLM (structured)")
        ai = await _retry_async(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:agent",
        )
        _log_info("LLM (agent) call succeeded")
        return ai

    # ---- Robust coercion: extract JSON from text, normalize, then instantiate ----

    @staticmethod
    def _coerce_to_discussion_topic(ai_content: str) -> Optional[DiscussionSummaryPerTopicSchema.DiscussionTopic]:
        """
        If the assistant content includes JSON (possibly fenced or embedded), extract,
        normalize (handles wrapped steps), and instantiate the nested Pydantic model.
        Return None if all attempts fail.
        """
        if not isinstance(ai_content, str) or not ai_content.strip():
            return None

        json_candidate = _extract_json_str(ai_content)
        if not json_candidate:
            return None

        try:
            obj = json.loads(json_candidate)
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None

        norm = _normalize_topic_obj(obj)
        if norm is None:
            return None

        try:
            model = DiscussionSummaryPerTopicSchema.DiscussionTopic(**norm)
            _log_info("Coercion succeeded (during pydantic model instantiation tests).")
            return model
        except Exception as exc:
            _log_warn(f"Coercion failed during pydantic model instantiation tests: {exc}")
            return None

    @staticmethod
    async def _invoke_structured(ai_content: str) -> DiscussionSummaryPerTopicSchema.DiscussionTopic:
        """
        Try coercion first (extract + normalize wrapped steps). If that fails,
        ask the structured LLM parser.
        """
        coerced = PerTopicDiscussionSummaryGenerationAgent._coerce_to_discussion_topic(ai_content)
        if coerced is not None:
            return coerced

        payload = [HumanMessage(content=ai_content or "")]

        async def _call():
            if hasattr(PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL, "ainvoke"):
                return await PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.ainvoke(payload)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, PerTopicDiscussionSummaryGenerationAgent._STRUCTURED_MODEL.invoke, payload
            )

        _log_info("Calling LLM (structured)")
        return await _retry_async(
            _call,
            retries=CFG.llm_retries,
            timeout_s=CFG.llm_timeout_s,
            backoff_base_s=CFG.llm_backoff_base_s,
            retry_reason="llm:structured",
        )

    # ----- Inner graph nodes (async) -----

    @staticmethod
    async def _agent_node(state: _PerTopicState):
        _log_tool_activity(state["messages"], ai_msg=None)
        ai = await PerTopicDiscussionSummaryGenerationAgent._invoke_agent(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    async def _respond_node(state: _PerTopicState):
        msgs = state["messages"]
        ai_content: Optional[str] = None

        # Prefer last assistant msg without tool calls
        for m in reversed(msgs):
            if getattr(m, "type", None) in ("ai", "assistant") and not getattr(m, "tool_calls", None):
                ai_content = m.content
                break
        # Fallback: any assistant msg
        if ai_content is None:
            for m in reversed(msgs):
                if getattr(m, "type", None) in ("ai", "assistant"):
                    ai_content = m.content
                    break
        # Final fallback: last message content
        if ai_content is None and msgs:
            ai_content = msgs[-1].content

        final_obj = await PerTopicDiscussionSummaryGenerationAgent._invoke_structured(ai_content or "")
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _PerTopicState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            _log_tool_activity(state["messages"], ai_msg=last)
            return "continue"
        return "respond"

    # ----- RunnableLambda wrappers (avoid staticmethod binding ambiguity) -----

    @staticmethod
    async def _agent_node_async(state: _PerTopicState):
        return await PerTopicDiscussionSummaryGenerationAgent._agent_node(state)

    @staticmethod
    async def _respond_node_async(state: _PerTopicState):
        return await PerTopicDiscussionSummaryGenerationAgent._respond_node(state)

    # ----- Compile inner ReAct graph -----

    @classmethod
    def _get_graph(cls):
        if cls._compiled_graph is not None:
            return cls._compiled_graph

        g = StateGraph(_PerTopicState)
        g.add_node("agent", RunnableLambda(cls._agent_node_async))
        g.add_node("respond", RunnableLambda(cls._respond_node_async))
        g.add_node("tools", ToolNode(cls.TOOLS, tags=["mongo-tools"]))
        g.set_entry_point("agent")
        g.add_conditional_edges("agent", cls._should_continue, {"continue": "tools", "respond": "respond"})
        g.add_edge("tools", "agent")
        g.add_edge("respond", END)

        cls._compiled_graph = g.compile()
        return cls._compiled_graph

    @staticmethod
    async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any], thread_id: str):
        class AtTemplate(Template):
            delimiter = "@"

        tpl = AtTemplate(DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT)
        sys_content = tpl.substitute(
            generated_summary=generated_summary_json,
            interview_topic=json.dumps(topic, ensure_ascii=False),
            thread_id=thread_id,
        )

        sys_msg = SystemMessage(content=sys_content)
        trigger = HumanMessage(content="Based on the provided instructions please start the process")

        graph = PerTopicDiscussionSummaryGenerationAgent._get_graph()
        result = await graph.ainvoke({"messages": [sys_msg, trigger]})

        # Graph may return a dict or the model directly
        if isinstance(result, dict):
            return result["final_response"]
        return result


# ==============================
# Outer graph: nodes & routing
# ==============================

class PerTopicDiscussionSummaryAgent:
    """Outer flow that runs inner per-topic loops concurrently and enforces policy."""

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Regenerate if the set of topics in the output does not exactly match the
        set of input topics. Includes a guard to avoid infinite loops if nothing
        was produced (single extra retry).
        """
        global _disc_retry_counter

        # Guard: if nothing produced, allow at most 1 retry under this condition
        if not getattr(state, "discussion_summary_per_topic", None) or \
           not getattr(state.discussion_summary_per_topic, "discussion_topics", None) or \
           len(state.discussion_summary_per_topic.discussion_topics) == 0:
            _log_retry("No discussion topics produced; retrying once", _disc_retry_counter)
            _disc_retry_counter += 1
            return _disc_retry_counter <= 2  # retry only once for the "no output" case

        input_topics = {t.topic for t in state.interview_topics.interview_topics}
        output_topics = {dt.topic for dt in state.discussion_summary_per_topic.discussion_topics}

        if input_topics != output_topics:
            missing = sorted(input_topics - output_topics)
            extra = sorted(output_topics - input_topics)
            _log_retry("Topic mismatch", _disc_retry_counter, {"missing": missing, "extra": extra})
            _disc_retry_counter += 1
            return True
        
        # Gated final-output logging
        rendered = _render_final_result(state.discussion_summary_per_topic.model_dump_json(indent=2))
        _log_info(f"Per-topic discussion summaries generated successfully | output={rendered}")

        return False

    @staticmethod
    async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Generate summaries for all topics concurrently:
          1) Normalize the input topics
          2) Serialize generated_summary for prompting
          3) Launch one inner-graph run per topic via asyncio.gather
          4) Force output topic names to match input names exactly
        """
        # Normalize topics list coming from parent state
        try:
            topics_list: List[Dict[str, Any]] = [t.model_dump() for t in state.interview_topics.interview_topics]
        except Exception:
            topics_list = state.interview_topics  # already a list[dict]

        if not isinstance(topics_list, list) or len(topics_list) == 0:
            raise ValueError("interview_topics must be a non-empty list[dict]")

        # Serialize generated_summary for prompt
        try:
            generated_summary_json = state.generated_summary.model_dump_json()
        except Exception:
            generated_summary_json = json.dumps(state.generated_summary, ensure_ascii=False)

        # Run all topic calls concurrently (each via inner graph)
        tasks = [
            asyncio.create_task(
                PerTopicDiscussionSummaryGenerationAgent._one_topic_call(generated_summary_json, topic, state.id)
            )
            for topic in topics_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful DiscussionTopic entries and enforce exact topic names
        discussion_topics: List[DiscussionSummaryPerTopicSchema.DiscussionTopic] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                _log_warn(f"Topic {idx} summarization failed: {result}")
                continue

            input_topic_name = (
                topics_list[idx].get("topic")
                or topics_list[idx].get("name")
                or topics_list[idx].get("title")
                or "Unknown"
            )

            # Coerce/update the topic name in the returned object
            try:
                if isinstance(result, dict):
                    result["topic"] = input_topic_name
                    result = DiscussionSummaryPerTopicSchema.DiscussionTopic(**result)
                elif hasattr(result, "model_copy"):  # pydantic v2
                    result = result.model_copy(update={"topic": input_topic_name})
                elif hasattr(result, "copy"):  # pydantic v1
                    result = result.copy(update={"topic": input_topic_name})
                else:
                    setattr(result, "topic", input_topic_name)

                discussion_topics.append(result)
            except Exception as exc:
                _log_warn(f"Failed to append structured response for topic index {idx}: {exc}")

        state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(discussion_topics=discussion_topics)

        # Gated final-output logging
        rendered = _render_final_result(state.discussion_summary_per_topic.model_dump_json(indent=2))
        _log_info(f"Per-topic discussion summaries generated before retry checks | output={rendered}")

        return state

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Topic-wise discussion summary graph:
        START -> discussion_summary_per_topic_generator
               -> (should_regenerate ? discussion_summary_per_topic_generator : END)
        """
        g = StateGraph(AgentInternalState)
        g.add_node(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryAgent.discussion_summary_per_topic_generator,
        )
        g.add_edge(START, "discussion_summary_per_topic_generator")
        g.add_conditional_edges(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryAgent.should_regenerate,
            {False: END, True: "discussion_summary_per_topic_generator"},
        )
        return g.compile(checkpointer=checkpointer, name="PerTopicDiscussionSummaryGenerationAgent")


if __name__ == "__main__":
    graph = PerTopicDiscussionSummaryAgent.get_graph()
    print(graph.get_graph().draw_ascii())
