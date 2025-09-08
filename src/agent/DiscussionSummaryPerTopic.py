# # Test code with one discussion function having loop every topic sequential running fine
# from __future__ import annotations
# from typing_extensions import TypedDict
# from langgraph.graph.message import add_messages
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import SystemMessage
# from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import CollectiveInterviewTopicSchema, DiscussionSummaryPerTopicSchema
# from ..prompt.discussion_summary_per_topic_generation_agent_prompt import DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT
# from ..model_handling import llm_tg
# from typing import Annotated, List, Dict, Any
# from operator import or_
# from langgraph.graph import StateGraph
# from langgraph.types import Send
# from langgraph.graph.message import add_messages
# import json

# set_llm_cache(InMemoryCache())


# class PerTopicDiscussionSummaryGenerationAgent:

#     llm_tg = llm_tg

#     # ---------- STATE ----------
#     class State(TypedDict):

#         messages: Annotated[list, add_messages]
#         # Input topics: list of dicts (or your pydantic TopicSchema converted to dict)
#         interview_topics: List[Dict[str, Any]]
#         # Single-topic slot used by worker (set during fan-out)
#         current_topic: Dict[str, Any] | None
#         generated_summary: Any
#         # Aggregated outputs: merge dicts from each parallel branch safely
#         outputs: Annotated[Dict[str, Any], or_]

#     # ---------- NODES ----------
#     # @staticmethod
#     # async def fan_out(state: AgentInternalState):
#     #     """Create one parallel branch per topic using Send(worker, payload)."""
#     #     sends: List[Send] = []
#     #     for topic in state["interview_topics"]:
#     #         payload: AgentInternalState = {
#     #             "messages": [],                     # reset messages for each branch
#     #             "interview_topics": state["interview_topics"],
#     #             "generated_summary": state["generated_summary"],
#     #             "current_topic": topic,             # pass one topic
#     #             "outputs": {},                      # worker will return {topic_name: response}
#     #         }
#     #         sends.append(Send("worker", payload))
#     #     return sends

#     @staticmethod
#     async def worker(state: State) -> State:
#         topic = state["current_topic"] or {}
#         topic_name = topic.get("topic", "unnamed-topic")
#         generated_summary = state.get("generated_summary", {})

#         # response = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
#         # .with_structured_output(CollectiveInterviewTopicSchema, method="function_calling") \
#         # .ainvoke(
#         #     [
#         #         SystemMessage(
#         #             content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#         #                 generated_summary=generated_summary.model_dump_json(),
#         #                 interview_topic=topic_name
#         #             )
#         #         ),
#         #         # state.messages[-1] if len(state.messages) else ""
#         #         state["messages"][-1] if len(state["messages"]) else ""
#         #     ]
#         # )
#         response = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
#         .with_structured_output(DiscussionSummaryPerTopicSchema.DiscussionTopic, method="function_calling") \
#         .ainvoke(
#             [
#                 SystemMessage(
#                     content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#                         generated_summary=generated_summary.model_dump_json(),
#                         interview_topic=json.dumps(topic)  # pass the full topic dict
#                     )
#                 ),
#                 state["messages"][-1] if len(state["messages"]) else ""
#             ]
#         )
#         state.interview_topics = response

#         # Merge into outputs under topic_name
#         return {
#             "outputs": {topic_name: response},
#             "generated_summary": state["generated_summary"],  # propagate
#             "interview_topics": state["interview_topics"],
#             "messages": state["messages"],
#             "current_topic": state["current_topic"],
#         }

#     @staticmethod
#     async def join(state: State) -> State:
#         """Optional: post-process after all branches merged."""
#         # e.g., compute total of questions, flatten, etc.
#         # Example: sum questions
#         outputs = state.get("outputs", {})
#         total = sum(v.get("total_questions", 0) for v in outputs.values())
#         # Optionally add a summary entry
#         outputs["_meta"] = {"total_questions": total, "topics_count": len(outputs) - ("_meta" in outputs)}
#         return {
#             "outputs": outputs,
#             "generated_summary": state["generated_summary"],  # propagate
#             "interview_topics": state["interview_topics"],
#             "messages": state["messages"],
#             "current_topic": state["current_topic"],
#         }

#     # Running
#     @staticmethod
#     async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
#         # Safety: ensure needed fields exist
#         if state.interview_topics is None or not state.interview_topics.interview_topics:
#             raise ValueError("No interview topics to summarize.")
#         if state.generated_summary is None:
#             raise ValueError("generated_summary is required.")

#         # We'll build a list of DiscussionSummaryPerTopicSchema.DiscussionTopic
#         discussion_topics: list[DiscussionSummaryPerTopicSchema.DiscussionTopic] = []

#         # Alias the nested class for structured output
#         TopicEntry = DiscussionSummaryPerTopicSchema.DiscussionTopic

#         for topic in state.interview_topics.interview_topics:
#             topic_name = topic.topic

#             # Serialize summary safely
#             try:
#                 summary_json = state.generated_summary.model_dump_json()
#             except Exception:
#                 summary_json = str(state.generated_summary)

#             # Ask the LLM to produce ONE discussion topic entry
#             resp = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
#                 .with_structured_output(TopicEntry, method="function_calling") \
#                 .ainvoke([
#                     SystemMessage(
#                         content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#                             generated_summary=summary_json,
#                             interview_topic=topic_name
#                         )
#                     )
#                 ])

#             discussion_topics.append(resp)

#         state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
#             discussion_topics=[dt.model_dump() if hasattr(dt, "model_dump") else dt for dt in discussion_topics]
#         )
#         return state

#     # Test
#     # @staticmethod
#     # async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
#     #     # Build subgraph
#     #     g = StateGraph(PerTopicDiscussionSummaryGenerationAgent.State)
#     #     g.add_node("fan_out", PerTopicDiscussionSummaryGenerationAgent.fan_out)
#     #     g.add_node("worker", PerTopicDiscussionSummaryGenerationAgent.worker)
#     #     g.add_node("join", PerTopicDiscussionSummaryGenerationAgent.join)
#     #     g.add_edge(START, "fan_out")
#     #     g.add_edge("worker", "join")
#     #     g.add_edge("join", END)
#     #     app = g.compile()

#     #     # Prepare initial subgraph state
#     #     initial: PerTopicDiscussionSummaryGenerationAgent.State = {
#     #         "messages": [],
#     #         "interview_topics": state.interview_topics.interview_topics,
#     #         "generated_summary": state.generated_summary,
#     #         "current_topic": None,
#     #         "outputs": {},
#     #     }

#     #     # IMPORTANT: await the subgraph
#     #     result = await app.ainvoke(initial)
#     #     # result["outputs"] is dict: topic_name -> DiscussionTopic (dict)

#     #     # Build top-level DiscussionSummaryPerTopicSchema
#     #     discussion_topics = list(result["outputs"].values())
#     #     dsp = DiscussionSummaryPerTopicSchema(discussion_topics=discussion_topics)

#     #     # Save back into the main AgentInternalState
#     #     state.discussion_summary_per_topic = dsp
#     #     return state



#     @staticmethod
#     async def fan_out(state: PerTopicDiscussionSummaryGenerationAgent.State):
#         for topic in state["interview_topics"]:
#             payload: PerTopicDiscussionSummaryGenerationAgent.State = {
#                 "messages": [],
#                 "interview_topics": state["interview_topics"],
#                 "generated_summary": state["generated_summary"],
#                 "current_topic": topic,
#                 "outputs": {},
#             }
#             yield Send("worker", payload)


#     @staticmethod
#     def get_graph(checkpointer=None):

#         graph_builder = StateGraph(
#             state_schema=AgentInternalState
#         )
        
#         graph_builder.add_node("discussion_summary_per_topic_generator", PerTopicDiscussionSummaryGenerationAgent.discussion_summary_per_topic_generator)
#         graph_builder.add_edge(START, "discussion_summary_per_topic_generator")
#         graph = graph_builder.compile(checkpointer=checkpointer, name="Discussion Summary per Topic Generation Agent")

#         return graph


# if __name__ == "__main__":
#     graph = PerTopicDiscussionSummaryGenerationAgent.get_graph()
#     g = graph.get_graph().draw_ascii()
#     print(g)


# Parallel topic running properly
import json
import asyncio
from typing import List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import DiscussionSummaryPerTopicSchema
from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
    DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT
)
from ..model_handling import llm_dts

set_llm_cache(InMemoryCache())


class PerTopicDiscussionSummaryGenerationAgent:        
    llm_dts = llm_dts

    @staticmethod
    async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any]):
        """Call LLM once for a single topic and return a structured DiscussionTopic."""
        TopicEntry = DiscussionSummaryPerTopicSchema.DiscussionTopic
        resp = await PerTopicDiscussionSummaryGenerationAgent.llm_dts \
            .with_structured_output(TopicEntry, method="function_calling") \
            .ainvoke([
                SystemMessage(
                    content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
                        generated_summary=generated_summary_json,
                        interview_topic=json.dumps(topic)
                    )
                )
            ])
        return resp

    @staticmethod
    async def should_regenerate(state: AgentInternalState):
        input_topics = {t.topic for t in state.interview_topics.interview_topics}
        output_topics = {dt.topic for dt in state.discussion_summary_per_topic.discussion_topics}
        if input_topics != output_topics:
            missing = input_topics - output_topics
            extra = output_topics - input_topics
            print(f"[PerTopic] Topic mismatch: missing {missing}, extra {extra}")
            return True
        else:
            return False

    @staticmethod
    async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Parallel per-topic summaries using asyncio.gather (no inner subgraph).
        Produces DiscussionSummaryPerTopicSchema with one entry per input topic.
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
            generated_summary_json = json.dumps(state.generated_summary)

        # Run all topic calls concurrently
        tasks = [
            asyncio.create_task(PerTopicDiscussionSummaryGenerationAgent._one_topic_call(generated_summary_json, topic))
            for topic in topics_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful DiscussionTopic entries
        discussion_topics = []
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                continue
            # r is already a structured DiscussionSummaryPerTopicSchema.DiscussionTopic
            try:
                discussion_topics.append(r)
            except Exception as e:
                print("Topic %d: could not append structured response (%s)", idx, e)

        state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
            discussion_topics=discussion_topics
        )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        gb = StateGraph(AgentInternalState)
        gb.add_node(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.discussion_summary_per_topic_generator
        )
        gb.add_edge(START, "discussion_summary_per_topic_generator")
        gb.add_conditional_edges(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.should_regenerate,
            {
                False: END,
                True: "discussion_summary_per_topic_generator"
            }
        )
        gb.add_edge("discussion_summary_per_topic_generator", END)
        return gb.compile(checkpointer=checkpointer, name="AgendaGenerationAgent")


if __name__ == "__main__":
    graph = PerTopicDiscussionSummaryGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())


# # Guard for all topic discussion summary generation try
# import json
# import asyncio
# from typing import List, Dict, Any, Tuple, Union

# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import SystemMessage
# from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache

# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import DiscussionSummaryPerTopicSchema
# from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
#     DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT
# )
# from ..model_handling import llm_dts

# set_llm_cache(InMemoryCache())


# def _as_dict(x: Any) -> Dict[str, Any]:
#     if hasattr(x, "model_dump"):
#         return x.model_dump()
#     if hasattr(x, "dict"):
#         return x.dict()
#     return x if isinstance(x, dict) else {}


# def _topic_label(d: Dict[str, Any]) -> str:
#     for k in ("topic", "name", "title", "label"):
#         v = d.get(k)
#         if isinstance(v, str) and v.strip():
#             return v.strip()
#     return "Unknown"


# class PerTopicDiscussionSummaryGenerationAgent:
#     llm_dts = llm_dts

#     @staticmethod
#     async def _one_topic_call(idx: int, generated_summary_json: str, topic_dict: Dict[str, Any]):
#         """Call LLM once for a single topic and return (idx, DiscussionTopic)."""
#         TopicEntry = DiscussionSummaryPerTopicSchema.DiscussionTopic
#         sysmsg = SystemMessage(
#             content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#                 generated_summary=generated_summary_json,
#                 interview_topic=json.dumps(topic_dict)
#             )
#         )
#         result = await PerTopicDiscussionSummaryGenerationAgent.llm_dts \
#             .with_structured_output(TopicEntry, method="function_calling") \
#             .ainvoke([sysmsg])
#         return idx, result

#     @staticmethod
#     async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
#         """
#         Generate per-topic discussion summaries with coverage check.
#         If some topics are missing/mismatched, regenerate only those until all are covered (or attempts exhausted).
#         """
#         # ---- Normalize topics list ----
#         try:
#             topics_list: List[Dict[str, Any]] = [t.model_dump() for t in state.interview_topics.interview_topics]
#         except Exception:
#             topics_list = list(state.interview_topics)  # assume list[dict]

#         if not topics_list:
#             raise ValueError("interview_topics must be a non-empty list.")

#         input_labels = [_topic_label(_as_dict(t)) for t in topics_list]
#         print(f"[PerTopic] Topics in: {input_labels} (count={len(input_labels)})")

#         # ---- Serialize global generated summary ----
#         try:
#             generated_summary_json = state.generated_summary.model_dump_json()
#         except Exception:
#             gs = state.generated_summary
#             generated_summary_json = json.dumps(gs.model_dump() if hasattr(gs, "model_dump") else gs)

#         n = len(topics_list)
#         ordered: List[DiscussionSummaryPerTopicSchema.DiscussionTopic] = [None] * n  # type: ignore

#         # ---- Regeneration loop: try to cover all topics ----
#         max_attempts = 3
#         concurrency = 2  # tweak if your provider allows more
#         attempt = 1

#         while attempt <= max_attempts:
#             # Figure out which indices still need generation
#             pending_indices = [i for i, x in enumerate(ordered) if x is None]
#             if not pending_indices:
#                 break  # all covered

#             if attempt == 1:
#                 print(f"[PerTopic] Generating all {n} topics (attempt {attempt}/{max_attempts})...")
#             else:
#                 pend_names = [input_labels[i] for i in pending_indices]
#                 print(f"[PerTopic] Regenerating {len(pending_indices)} pending topics "
#                       f"(attempt {attempt}/{max_attempts}): {pend_names}")

#             sem = asyncio.Semaphore(concurrency)

#             async def _guarded(i: int) -> Tuple[int, Union[Exception, Any]]:
#                 async with sem:
#                     topic_dict = topics_list[i]
#                     try:
#                         return await asyncio.wait_for(
#                             PerTopicDiscussionSummaryGenerationAgent._one_topic_call(
#                                 i, generated_summary_json, topic_dict
#                             ),
#                             timeout=60,
#                         )
#                     except Exception as e:
#                         print(f"[PerTopic][ERROR] idx={i} topic='{input_labels[i]}' failed: "
#                               f"{type(e).__name__}: {e}")
#                         return i, e

#             tasks = [_guarded(i) for i in pending_indices]
#             results = await asyncio.gather(*tasks, return_exceptions=False)

#             # Validate & place results
#             for i, payload in results:
#                 if isinstance(payload, Exception):
#                     continue
#                 output_label = _topic_label(_as_dict(payload))
#                 expected_label = input_labels[i]
#                 if output_label != expected_label:
#                     print(f"[PerTopic][WARN] Label mismatch at idx={i}: "
#                           f"input='{expected_label}' vs output='{output_label}' (will retry this topic)")
#                     continue
#                 ordered[i] = payload  # success

#             attempt += 1

#         # ---- Final coverage check ----
#         still_missing = [i for i, x in enumerate(ordered) if x is None]
#         if still_missing:
#             missing_names = [input_labels[i] for i in still_missing]
#             raise RuntimeError(
#                 f"[PerTopic][ERROR] Could not generate discussion summaries for indices {still_missing} "
#                 f"(topics: {missing_names}) after {max_attempts} attempts."
#             )

#         print(f"[PerTopic] Successfully generated coverage for {n}/{n} topics.")
#         state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
#             discussion_topics=ordered
#         )
#         return state

#     @staticmethod
#     def get_graph(checkpointer=None):
#         gb = StateGraph(AgentInternalState)
#         gb.add_node(
#             "discussion_summary_per_topic_generator",
#             PerTopicDiscussionSummaryGenerationAgent.discussion_summary_per_topic_generator
#         )
#         gb.add_edge(START, "discussion_summary_per_topic_generator")
#         gb.add_edge("discussion_summary_per_topic_generator", END)
#         return gb.compile(checkpointer=checkpointer, name="PerTopicDiscussionSummaryGenerationAgent")


# if __name__ == "__main__":
#     graph = PerTopicDiscussionSummaryGenerationAgent.get_graph()
#     print(graph.get_graph().draw_ascii())