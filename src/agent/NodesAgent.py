# Running Test with all topics given
# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import SystemMessage
# from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import (
#     NodesSchema,
#     TopicWithNodesSchema,
# )
# from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
# from ..model_handling import llm_dts
# from typing import List, Dict, Any
# import json

# set_llm_cache(InMemoryCache())


# class NodesGenerationAgent:

#     llm_dts = llm_dts

#     @staticmethod
#     async def nodes_generator(state: AgentInternalState) -> AgentInternalState:
#         # Safety checks
#         if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
#             raise ValueError("No interview topics to summarize.")
#         if state.generated_summary is None:
#             raise ValueError("generated_summary is required.")
#         if state.discussion_summary_per_topic is None:
#             raise ValueError("discussion_summary_per_topic is required.")

#         # Serialize inputs for the prompt
#         try:
#             discussion_summary_json = state.discussion_summary_per_topic.model_dump_json()
#         except Exception:
#             discussion_summary_json = json.dumps(state.discussion_summary_per_topic.discussion_topics)

#         topics_out: List[TopicWithNodesSchema] = []

#         # For each topic, ask the model to produce nodes for that topic
#         for topic in state.interview_topics.interview_topics:
#             # total questions budget for this topic
#             total_no_questions_context = getattr(topic, "total_questions", None)
#             interview_topic_json = topic.model_dump_json() if hasattr(topic, "model_dump_json") else json.dumps(topic)
#             resp: TopicWithNodesSchema = await NodesGenerationAgent.llm_dts \
#                 .with_structured_output(TopicWithNodesSchema, method="function_calling") \
#                 .ainvoke([
#                     SystemMessage(
#                         content=NODES_AGENT_PROMPT.format(
#                             discussion_summary=discussion_summary_json,
#                             # interview_topic_json = interview_topic_json,
#                             total_no_questions_context=total_no_questions_context
#                         )
#                     )
#                 ])

#             # Ensure the topic name is present; if your prompt already includes it, this is redundant
#             if not resp.topic:
#                 resp = TopicWithNodesSchema(topic=getattr(topic, "topic", "Unknown"), nodes=resp.nodes)

#             topics_out.append(resp)

#         # Assign back to state using the CORRECT container field
#         state.nodes = NodesSchema(
#             topics_with_nodes=[t.model_dump() if hasattr(t, "model_dump") else t for t in topics_out]
#         )
#         return state

#     @staticmethod
#     def get_graph(checkpointer=None):
#         graph_builder = StateGraph(state_schema=AgentInternalState)
#         graph_builder.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
#         graph_builder.add_edge(START, "nodes_generator")
#         graph = graph_builder.compile(checkpointer=checkpointer, name="Nodes Generation Agent")
#         return graph

# if __name__ == "__main__":
#     graph = NodesGenerationAgent.get_graph()
#     g = graph.get_graph().draw_ascii()
#     print(g)

from typing import List, Any, Dict
from langgraph.graph import StateGraph, START, END
import json
import copy
from langchain_core.messages import SystemMessage
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import NodesSchema, TopicWithNodesSchema
from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
from ..model_handling import llm_dts

ALLOWED_TYPES = {"Direct", "Deep Dive"}


class NodesGenerationAgent:
    llm_dts = llm_dts

    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "dict"):
            return x.dict()
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _get_topic_name(obj: Any) -> str:
        d = NodesGenerationAgent._as_dict(obj)
        for k in ("topic", "name", "title", "label"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"

    @staticmethod
    def _get_total_questions(topic_obj: Any, dspt_obj: Any) -> int:
        # prefer topic.total_questions, then dspt.total_questions
        for src in (topic_obj, dspt_obj):
            d = NodesGenerationAgent._as_dict(src)
            tq = d.get("total_questions")
            if isinstance(tq, int) and tq >= 2:
                return tq
        raise ValueError("total_questions must be >= 2 for each topic")

    @staticmethod
    def _to_json_one(x: Any) -> str:
        # deep copy -> dict -> json (prevents accidental upstream mutation)
        if hasattr(x, "model_dump"):
            return json.dumps(copy.deepcopy(x.model_dump()))
        if hasattr(x, "dict"):
            return json.dumps(copy.deepcopy(x.dict()))
        return json.dumps(copy.deepcopy(x))

    @staticmethod
    def _validate_topic_nodes(resp: TopicWithNodesSchema, expected_topic: str, T: int) -> None:
        if resp.topic != expected_topic:
            raise ValueError(f"Topic label mismatch: expected '{expected_topic}', got '{resp.topic}'")

        nodes = resp.nodes or []
        if len(nodes) != T:
            raise ValueError(f"Expected exactly {T} nodes, got {len(nodes)}")

        for i, n in enumerate(nodes, start=1):
            # ids and linear next pointers
            if n.id != i:
                raise ValueError("ids must be 1..T increasing by 1")
            if i < T and n.next_node != i + 1:
                raise ValueError("next_node must be i+1 for nodes before last")
            if i == T and n.next_node is not None:
                raise ValueError("last node must have next_node=null")

            # types & grading per position
            if n.question_type not in ALLOWED_TYPES:
                raise ValueError(f"Invalid question_type '{n.question_type}' (allowed: Direct, Deep Dive)")

            if i == 1:
                if not (n.question_type == "Direct" and n.graded is False):
                    raise ValueError("Node#1 must be Direct and graded=false")
                if n.total_question_threshold is not None or n.question_guidelines is not None:
                    raise ValueError("Direct nodes must have null guidelines/threshold")
            elif i < T:
                if not (n.question_type == "Direct" and n.graded is True):
                    raise ValueError("Intermediate nodes must be Direct and graded=true")
                if n.total_question_threshold is not None or n.question_guidelines is not None:
                    raise ValueError("Direct nodes must have null guidelines/threshold")
            else:
                if not (n.question_type == "Deep Dive" and n.graded is True):
                    raise ValueError("Last node must be Deep Dive and graded=true")
                if not isinstance(n.total_question_threshold, int) or n.total_question_threshold < 1:
                    raise ValueError("Deep Dive requires total_question_threshold >= 1")
                if not n.question_guidelines or not str(n.question_guidelines).strip():
                    raise ValueError("Deep Dive requires non-empty question_guidelines")

            # skills required
            if not n.skills or not all(isinstance(s, str) and s.strip() for s in n.skills):
                raise ValueError("Each node must list ≥1 non-empty skill")

    @staticmethod
    async def nodes_generator(state: AgentInternalState) -> AgentInternalState:
        # Guards
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")

        topics_list = list(state.interview_topics.interview_topics)
        try:
            summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
        except Exception:
            summaries_list = list(state.discussion_summary_per_topic)

        pair_count = min(len(topics_list), len(summaries_list))
        if pair_count == 0:
            raise ValueError("No topics/summaries to process.")

        # Snapshot the summary BEFORE processing (sanity: ensure we don't mutate upstream)
        snapshot = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True
        )

        out: List[TopicWithNodesSchema] = []

        for topic_obj, dspt_obj in zip(topics_list, summaries_list):
            T = NodesGenerationAgent._get_total_questions(topic_obj, dspt_obj)
            per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)
            expected_topic = NodesGenerationAgent._get_topic_name(dspt_obj)

            # Call LLM with ONLY this topic’s summary + T
            resp: TopicWithNodesSchema = await NodesGenerationAgent.llm_dts \
                .with_structured_output(TopicWithNodesSchema, method="function_calling") \
                .ainvoke([
                    SystemMessage(content=NODES_AGENT_PROMPT.format(
                        per_topic_summary_json=per_topic_summary_json,
                        total_no_questions_context=T
                    ))
                ])

            # Strict validation (no padding, no trimming, no repair)
            NodesGenerationAgent._validate_topic_nodes(resp, expected_topic, T)
            out.append(resp)

        # Sanity: verify upstream summary wasn’t mutated
        after = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True
        )
        if after != snapshot:
            raise RuntimeError("discussion_summary_per_topic mutated during node generation")

        state.nodes = NodesSchema(
            topics_with_nodes=[t.model_dump() if hasattr(t, "model_dump") else t for t in out]
        )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
        graph_builder.add_edge(START, "nodes_generator")
        graph = graph_builder.compile(checkpointer=checkpointer, name="Nodes Generation Agent")
        return graph


if __name__ == "__main__":
    graph = NodesGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)


# # Test per topic node
# import json
# import asyncio
# from typing import List, Dict, Any

# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import SystemMessage
# from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache

# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import NodesPerTopicSchema
# from ..prompt.nodes_agent_prompt import (
#     NODES_AGENT_PROMPT
# )
# from ..model_handling import llm_dts

# set_llm_cache(InMemoryCache())


# class NodesAgent:
#     llm_dts = llm_dts

#     @staticmethod
#     async def _one_topic_call(discussion_summary_json: str, topic: Dict[str, Any]):
#         """Call LLM once for a single topic and return a structured DiscussionTopic."""
#         TopicEntry = NodesPerTopicSchema.DiscussionTopic
#         resp = await NodesAgent.llm_dts \
#             .with_structured_output(TopicEntry, method="function_calling") \
#             .ainvoke([
#                 SystemMessage(
#                     content=NODES_AGENT_PROMPT.format(
#                         discussion_summary=discussion_summary_json,
#                         total_no_questions_context=json.dumps(topic)["total_questions"]
#                     )
#                 )
#             ])
#         return resp

#     @staticmethod
#     async def nodes_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
#         """
#         Parallel per-topic nodes using asyncio.gather (no inner subgraph).
#         Produces NodesPerTopicSchema with one entry per input topic.
#         """
#         # Normalize topics list coming from parent state
#         try:
#             topics_list: List[Dict[str, Any]] = [t.model_dump() for t in state.interview_topics.interview_topics]
#         except Exception:
#             topics_list = state.interview_topics  # already a list[dict]

#         if not isinstance(topics_list, list) or len(topics_list) == 0:
#             raise ValueError("interview_topics must be a non-empty list[dict]")

#         # Serialize generated_node for prompt
#         try:
#             generated_node_json = state.generated_node.model_dump_json()
#         except Exception:
#             generated_node_json = json.dumps(state.generated_node)

#         # Run all topic calls concurrently
#         tasks = [
#             NodesAgent._one_topic_call(generated_node_json, topic)
#             for topic in topics_list
#         ]
#         results = await asyncio.gather(*tasks, return_exceptions=True)

#         # Collect successful DiscussionTopic entries
#         discussion_topics = []
#         for idx, r in enumerate(results):
#             if isinstance(r, Exception):
#                 continue
#             # r is already a structured NodesPerTopicSchema.DiscussionTopic
#             try:
#                 discussion_topics.append(r)
#             except Exception as e:
#                 print("Topic %d: could not append structured response (%s)", idx, e)

#         state.nodes_per_topic = NodesPerTopicSchema(
#             discussion_topics=discussion_topics
#         )
#         return state

#     @staticmethod
#     def get_graph(checkpointer=None):
#         gb = StateGraph(AgentInternalState)
#         gb.add_node(
#             "nodes_per_topic_generator",
#             NodesAgent.nodes_per_topic_generator
#         )
#         gb.add_edge(START, "nodes_per_topic_generator")
#         gb.add_edge("nodes_per_topic_generator", END)
#         return gb.compile(checkpointer=checkpointer, name="AgendaGenerationAgent")


# if __name__ == "__main__":
#     graph = NodesAgent.get_graph()
#     print(graph.get_graph().draw_ascii())
