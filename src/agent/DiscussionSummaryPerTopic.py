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
#     # @staticmethod
#     # async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
#     #     # Safety: ensure needed fields exist
#     #     if state.interview_topics is None or not state.interview_topics.interview_topics:
#     #         raise ValueError("No interview topics to summarize.")
#     #     if state.generated_summary is None:
#     #         raise ValueError("generated_summary is required.")

#     #     # We'll build a list of DiscussionSummaryPerTopicSchema.DiscussionTopic
#     #     discussion_topics: list[DiscussionSummaryPerTopicSchema.DiscussionTopic] = []

#     #     # Alias the nested class for structured output
#     #     TopicEntry = DiscussionSummaryPerTopicSchema.DiscussionTopic

#     #     for topic in state.interview_topics.interview_topics:
#     #         topic_name = topic.topic

#     #         # Serialize summary safely
#     #         try:
#     #             summary_json = state.generated_summary.model_dump_json()
#     #         except Exception:
#     #             summary_json = str(state.generated_summary)

#     #         # Ask the LLM to produce ONE discussion topic entry
#     #         resp = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
#     #             .with_structured_output(TopicEntry, method="function_calling") \
#     #             .ainvoke([
#     #                 SystemMessage(
#     #                     content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#     #                         generated_summary=summary_json,
#     #                         interview_topic=topic_name
#     #                     )
#     #                 )
#     #             ])

#     #         discussion_topics.append(resp)

#     #     state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
#     #         discussion_topics=[dt.model_dump() if hasattr(dt, "model_dump") else dt for dt in discussion_topics]
#     #     )
#     #     return state

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
#     async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
        
#         class State(TypedDict):
#             topic: str
#             combined_output: str
#         # Nodes
#         async def call_llm_1(state: State):
#             msg = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
#             .with_structured_output(DiscussionSummaryPerTopicSchema.DiscussionTopic, method="function_calling") \
#             .ainvoke(
#                 [
#                     SystemMessage(
#                         content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#                             generated_summary=state.generated_summary.model_dump_json(),
#                             interview_topic=json.dumps(topic)  # pass the full topic dict
#                         )
#                     ),
#                     state["messages"][-1] if len(state["messages"]) else ""
#                 ]
#             )

#             return {"topic1": msg.content}


#         async def call_llm_2(state: State):
#             msg = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
#             .with_structured_output(DiscussionSummaryPerTopicSchema.DiscussionTopic, method="function_calling") \
#             .ainvoke(
#                 [
#                     SystemMessage(
#                         content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#                             generated_summary=state.generated_summary.model_dump_json(),
#                             interview_topic=json.dumps(topic)  # pass the full topic dict
#                         )
#                     ),
#                     state["messages"][-1] if len(state["messages"]) else ""
#                 ]
#             )

#             return {"topic2": msg.content}


#         async def call_llm_3(state: State):
#             msg = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
#             .with_structured_output(DiscussionSummaryPerTopicSchema.DiscussionTopic, method="function_calling") \
#             .ainvoke(
#                 [
#                     SystemMessage(
#                         content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
#                             generated_summary=generated_summary.model_dump_json(),
#                             interview_topic=json.dumps(topic)  # pass the full topic dict
#                         )
#                     ),
#                     state["messages"][-1] if len(state["messages"]) else ""
#                 ]
#             )

#             return {"topic3": msg.content}

#         def aggregator(state: State):
#             """Combine into a single output"""

#             combined += f"TOPIC 1:\n{state['topic1']}\n\n"
#             combined += f"TOPIC 2:\n{state['topic2']}\n\n"
#             combined += f"TOPIC 3:\n{state['topic3']}\n\n"
#             return {"combined_output": combined}


#         # Build workflow
#         parallel_builder = StateGraph(State)

#         # Add nodes
#         parallel_builder.add_node("call_llm_1", call_llm_1)
#         parallel_builder.add_node("call_llm_2", call_llm_2)
#         parallel_builder.add_node("call_llm_3", call_llm_3)
#         parallel_builder.add_node("aggregator", aggregator)

#         # Add edges to connect nodes
#         parallel_builder.add_edge(START, "call_llm_1")
#         parallel_builder.add_edge(START, "call_llm_2")
#         parallel_builder.add_edge(START, "call_llm_3")
#         parallel_builder.add_edge("call_llm_1", "aggregator")
#         parallel_builder.add_edge("call_llm_2", "aggregator")
#         parallel_builder.add_edge("call_llm_3", "aggregator")
#         parallel_builder.add_edge("aggregator", END)
#         # Build subgraph
#         g = StateGraph(PerTopicDiscussionSummaryGenerationAgent.State)
#         g.add_node("fan_out", PerTopicDiscussionSummaryGenerationAgent.fan_out)
#         g.add_node("worker", PerTopicDiscussionSummaryGenerationAgent.worker)
#         g.add_node("join", PerTopicDiscussionSummaryGenerationAgent.join)
#         g.add_edge(START, "fan_out")
#         g.add_edge("worker", "join")
#         g.add_edge("join", END)
#         app = g.compile()

#         # Prepare initial subgraph state
#         # Use the list of dicts for interview_topics
#         initial: PerTopicDiscussionSummaryGenerationAgent.State = {
#             "messages": [],
#             "interview_topics": [t.model_dump() for t in state.interview_topics.interview_topics],
#             "generated_summary": state.generated_summary,
#             "current_topic": None,
#             "outputs": {},
#         }

#         # Await the subgraph
#         result = await app.ainvoke(initial)
#         # result["outputs"] is dict: topic_name -> DiscussionTopic (dict)

#         # Remove any _meta key if present
#         discussion_topics = [v for k, v in result["outputs"].items() if k != "_meta"]
#         dsp = DiscussionSummaryPerTopicSchema(discussion_topics=discussion_topics)

#         # Save back into the main AgentInternalState
#         state.discussion_summary_per_topic = dsp
#         return state

#     # @staticmethod
#     # async def should_continue(state: AgentInternalState) -> bool:
#     #     if state.listoftopics_discussion_summary != []:
#     #         return True
#     #     return False

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


# per_topic_agent.py
# from __future__ import annotations
import json
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from operator import or_

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import (
    CollectiveInterviewTopicSchema, 
    DiscussionSummaryPerTopicSchema
)
from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
    DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT
)
from ..model_handling import llm_tg

set_llm_cache(InMemoryCache())


class PerTopicDiscussionSummaryGenerationAgent:
    llm_tg = llm_tg

    class FanoutState(TypedDict):
        # Subgraph state used only inside the per-topic fanout
        messages: Annotated[list, add_messages]
        interview_topics: List[Dict[str, Any]]         # list of topic dicts
        current_topic: Optional[Dict[str, Any]]        # topic dict for this branch
        generated_summary: Any                         # your upstream summary object
        outputs: Annotated[Dict[str, Any], or_]        # merged results { topic_name: result }

    @staticmethod
    async def fan_out(state: "PerTopicDiscussionSummaryGenerationAgent.FanoutState"):
        """Yield a Send for each topic; LangGraph will process workers concurrently."""
        for topic in state["interview_topics"]:
            payload: PerTopicDiscussionSummaryGenerationAgent.FanoutState = {
                "messages": [],                           # fresh message list per branch
                "interview_topics": state["interview_topics"],
                "generated_summary": state["generated_summary"],
                "current_topic": topic,
                "outputs": {},                            # worker returns per-topic dict
            }
            # IMPORTANT: yield a Send to fan out
            yield Send("worker", payload)

    @staticmethod
    async def worker(state: "PerTopicDiscussionSummaryGenerationAgent.FanoutState") -> "PerTopicDiscussionSummaryGenerationAgent.FanoutState":
        """One branch per topic: call the LLM and return structured output keyed by topic name."""
        topic: Dict[str, Any] = state.get("current_topic") or {}
        topic_name = topic.get("topic", "unnamed-topic")
        generated_summary = state.get("generated_summary")

        # Safely serialize your upstream summary for the prompt
        try:
            summary_json = generated_summary.model_dump_json()
        except Exception:
            # fallback if it's already a dict/str
            summary_json = getattr(generated_summary, "json", lambda: str(generated_summary))()

        # Ask the model for ONE structured entry for this topic
        TopicEntry = DiscussionSummaryPerTopicSchema.DiscussionTopic
        response = await PerTopicDiscussionSummaryGenerationAgent.llm_tg \
            .with_structured_output(TopicEntry, method="function_calling") \
            .ainvoke([
                SystemMessage(
                    content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
                        generated_summary=summary_json,
                        interview_topic=json.dumps(topic)  # pass full dict
                    )
                )
            ])

        # Merge into outputs under the topic name. DO NOT mutate interview_topics.
        return {
            "outputs": {topic_name: response},
            "generated_summary": generated_summary,
            "interview_topics": state["interview_topics"],
            "messages": state["messages"],
            "current_topic": state["current_topic"],
        }

    @staticmethod
    async def join(state: "PerTopicDiscussionSummaryGenerationAgent.FanoutState") -> "PerTopicDiscussionSummaryGenerationAgent.FanoutState":
        """Optional post-processing: add meta, totals, etc."""
        outputs = state.get("outputs", {})
        # Example meta
        meta = {
            "topics_count": len([k for k in outputs.keys() if k != "_meta"])
        }
        outputs["_meta"] = meta
        return {
            "outputs": outputs,
            "generated_summary": state["generated_summary"],
            "interview_topics": state["interview_topics"],
            "messages": state["messages"],
            "current_topic": state["current_topic"],
        }

    @staticmethod
    async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Builds a subgraph that runs one 'worker' per topic in parallel,
        then merges to a single DiscussionSummaryPerTopicSchema.
        """
        # Build parallel subgraph
        g = StateGraph(PerTopicDiscussionSummaryGenerationAgent.FanoutState)
        g.add_node("fan_out", PerTopicDiscussionSummaryGenerationAgent.fan_out)
        g.add_node("worker", PerTopicDiscussionSummaryGenerationAgent.worker)
        g.add_node("join", PerTopicDiscussionSummaryGenerationAgent.join)

        g.add_edge(START, "fan_out")
        # Each 'Send("worker",...)" returns here; edges from worker to join accumulate
        g.add_edge("worker", "join")
        g.add_edge("join", END)

        app = g.compile(name="PerTopicParallelSummaries")

        # Prepare initial fanout state from your top-level AgentInternalState
        # Ensure state.interview_topics is a list[dict]
        try:
            topics_list = [t.model_dump() for t in state.interview_topics.interview_topics]
        except Exception:
            topics_list = state.interview_topics  # already list[dict]?

        initial: PerTopicDiscussionSummaryGenerationAgent.FanoutState = {
            "messages": [],                 # we are not using messages per-branch here
            "interview_topics": topics_list,
            "generated_summary": state.generated_summary,
            "current_topic": None,
            "outputs": {},
        }

        # Run the subgraph (branches execute concurrently under the hood via asyncio)
        result = await app.ainvoke(initial)

        # Convert outputs dict -> list[DiscussionTopic] for your top-level schema
        # Skip _meta
        per_topic = [v for k, v in result["outputs"].items() if k != "_meta"]
        dsp = DiscussionSummaryPerTopicSchema(discussion_topics=per_topic)

        # Save back to your main state
        state.discussion_summary_per_topic = dsp
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        """
        Top-level graph that has a single node invoking the parallel per-topic subgraph.
        You can add more nodes before/after as needed.
        """
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.discussion_summary_per_topic_generator
        )
        graph_builder.add_edge(START, "discussion_summary_per_topic_generator")
        graph = graph_builder.compile(
            checkpointer=checkpointer,
            name="Discussion Summary per Topic Generation Agent"
        )
        return graph
    
if __name__ == "__main__":
    graph = PerTopicDiscussionSummaryGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
