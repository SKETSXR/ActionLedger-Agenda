# Running with schema check and regenerate
from typing import List, Any, Dict
from langgraph.graph import StateGraph, START, END
import json, copy
from langchain_core.messages import SystemMessage
from pydantic import ValidationError
from src.mongo_tools import get_mongo_tools
from langgraph.prebuilt import ToolNode
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import NodesSchema, TopicWithNodesSchema
from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
from ..model_handling import llm_n


class NodesGenerationAgent:
    llm_n = llm_n
    MONGO_TOOLS = get_mongo_tools(llm=llm_n)
    # mongo_toolnode = ToolNode(MONGO_TOOLS)
    # llm_n_with_tools = llm_n.bind_tools(mongo_toolnode)
    llm_n_with_tools = llm_n.bind_tools(MONGO_TOOLS) 

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
    def _to_json_one(x: Any) -> str:
        # deep copy -> dict -> json (avoid mutating upstream structures)
        if hasattr(x, "model_dump"):
            return json.dumps(copy.deepcopy(x.model_dump()))
        if hasattr(x, "dict"):
            return json.dumps(copy.deepcopy(x.dict()))
        return json.dumps(copy.deepcopy(x))

    @staticmethod
    def _get_total_questions(topic_obj: Any, dspt_obj: Any) -> int:

        for src in (topic_obj, dspt_obj):
            d = NodesGenerationAgent._as_dict(src)
            tq = d.get("total_questions")
            if isinstance(tq, int) and tq >= 2:
                return tq
        raise ValueError("total_questions must be >= 2 for each topic")

    @staticmethod
    async def _gen_once(per_topic_summary_json: str, T, nodes_error) -> TopicWithNodesSchema:
        sys = NODES_AGENT_PROMPT.format(
            per_topic_summary_json=per_topic_summary_json,
            total_no_questions_context=T,
            nodes_error=nodes_error
        )
        return await NodesGenerationAgent.llm_n_with_tools \
            .with_structured_output(TopicWithNodesSchema, method="function_calling") \
            .ainvoke([SystemMessage(content=sys)])

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Return True if we need to regenerate (schema invalid), else False.
        Validates the container (NodesSchema) and each TopicWithNodesSchema item inside it.
        """
        # Nothing produced yet? -> regenerate
        if getattr(state, "nodes", None) is None:
            return True

        # Validate NodesSchema (container)
        try:
            # accept either a Pydantic instance or a plain dictionary
            NodesSchema.model_validate(
                state.nodes.model_dump() if hasattr(state.nodes, "model_dump") else state.nodes
            )
        except ValidationError as ve:
            print("[NodesGen][ValidationError] Container NodesSchema invalid")
            print(str(ve))
            state.nodes_error += "The previous generated o/p did not follow the given schema as it got following errors:\n" + (getattr(state, "nodes_error", "") or "") + \
                                "\n[NodesSchema ValidationError]\n" + str(ve) + "\n"
            return True

        # Validate each topic payload
        try:
            topics_payload = (
                state.nodes.topics_with_nodes
                if hasattr(state.nodes, "topics_with_nodes")
                else state.nodes.get("topics_with_nodes", [])
            )
        except Exception as e:
            print("[NodesGen][ValidationError] Could not read topics_with_nodes:", e)
            state.nodes_error += "\n[NodesSchema Payload Error]\n" + str(e) + "\n"
            return True

        any_invalid = False
        for idx, item in enumerate(topics_payload):
            try:
                # item can be pydantic model or dictionary
                TopicWithNodesSchema.model_validate(
                    item.model_dump() if hasattr(item, "model_dump") else item
                )
            except ValidationError as ve:
                any_invalid = True
                print(f"[NodesGen][ValidationError] TopicWithNodesSchema invalid at index {idx}")
                print(str(ve))
                state.nodes_error += f"\n[TopicWithNodesSchema ValidationError idx={idx}]\n" + str(ve) + "\n"

        return any_invalid

    # -----------------------------
    # Main Node generation node
    # -----------------------------
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

        # Snapshot upstream summaries (sanity: ensure we don't mutate them)
        snapshot = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True
        )

        out: List[TopicWithNodesSchema] = []

        for topic_obj, dspt_obj in zip(topics_list, summaries_list):
            T = NodesGenerationAgent._get_total_questions(topic_obj, dspt_obj)
            per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)

            resp = await NodesGenerationAgent._gen_once(per_topic_summary_json, T, state.nodes_error)
            out.append(resp)

        # Verify upstream summary wasn’t mutated
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
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
        gb.add_edge(START, "nodes_generator")
        gb.add_conditional_edges(
            "nodes_generator",
            NodesGenerationAgent.should_regenerate,  # returns True/False
            { True: "nodes_generator", False: END }
        )
        return gb.compile(checkpointer=checkpointer, name="Nodes Generation Agent")

if __name__ == "__main__":
    graph = NodesGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)

# # Running nodes agent without schema check
# from typing import List, Any, Dict
# from langgraph.graph import StateGraph, START
# import json
# import copy
# from langchain_core.messages import SystemMessage
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import NodesSchema, TopicWithNodesSchema
# from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
# from ..model_handling import llm_n

# ALLOWED_TYPES = {"Direct", "Deep Dive"}


# class NodesGenerationAgent:
#     llm_n = llm_n

#     @staticmethod
#     def _as_dict(x: Any) -> Dict[str, Any]:
#         if hasattr(x, "model_dump"):
#             return x.model_dump()
#         if hasattr(x, "dict"):
#             return x.dict()
#         return x if isinstance(x, dict) else {}

#     @staticmethod
#     def _get_topic_name(obj: Any) -> str:
#         d = NodesGenerationAgent._as_dict(obj)
#         for k in ("topic", "name", "title", "label"):
#             v = d.get(k)
#             if isinstance(v, str) and v.strip():
#                 return v.strip()
#         return "Unknown"

#     @staticmethod
#     def _get_total_questions(topic_obj: Any, dspt_obj: Any) -> int:

#         for src in (topic_obj, dspt_obj):
#             d = NodesGenerationAgent._as_dict(src)
#             tq = d.get("total_questions")
#             if isinstance(tq, int) and tq >= 2:
#                 return tq
#         raise ValueError("total_questions must be >= 2 for each topic")

#     @staticmethod
#     def _to_json_one(x: Any) -> str:
#         # deep copy -> dict -> json (prevents accidental upstream mutation)
#         if hasattr(x, "model_dump"):
#             return json.dumps(copy.deepcopy(x.model_dump()))
#         if hasattr(x, "dict"):
#             return json.dumps(copy.deepcopy(x.dict()))
#         return json.dumps(copy.deepcopy(x))

#     @staticmethod
#     async def nodes_generator(state: AgentInternalState) -> AgentInternalState:
#         # Guards
#         if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
#             raise ValueError("No interview topics to summarize.")
#         if state.discussion_summary_per_topic is None:
#             raise ValueError("discussion_summary_per_topic is required.")

#         topics_list = list(state.interview_topics.interview_topics)
#         try:
#             summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
#         except Exception:
#             summaries_list = list(state.discussion_summary_per_topic)

#         pair_count = min(len(topics_list), len(summaries_list))
#         if pair_count == 0:
#             raise ValueError("No topics/summaries to process.")

#         # Snapshot the summary BEFORE processing (sanity: ensure we don't mutate upstream)
#         snapshot = json.dumps(
#             [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
#             sort_keys=True
#         )

#         out: List[TopicWithNodesSchema] = []

#         for topic_obj, dspt_obj in zip(topics_list, summaries_list):
#             T = NodesGenerationAgent._get_total_questions(topic_obj, dspt_obj)
#             per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)
#             expected_topic = NodesGenerationAgent._get_topic_name(dspt_obj)

#             # Call LLM with only this topic’s summary + the topic question count (T)
#             resp: TopicWithNodesSchema = await NodesGenerationAgent.llm_n \
#                 .with_structured_output(TopicWithNodesSchema, method="function_calling") \
#                 .ainvoke([
#                     SystemMessage(content=NODES_AGENT_PROMPT.format(
#                         per_topic_summary_json=per_topic_summary_json,
#                         total_no_questions_context=T
#                     ))
#                 ])

#             # Strict validation (no padding, no trimming, no repair)
#             # NodesGenerationAgent._validate_topic_nodes(resp, expected_topic, T)
#             out.append(resp)

#         # Sanity: verify upstream summary wasn’t mutated
#         after = json.dumps(
#             [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
#             sort_keys=True
#         )
#         if after != snapshot:
#             raise RuntimeError("discussion_summary_per_topic mutated during node generation")

#         state.nodes = NodesSchema(
#             topics_with_nodes=[t.model_dump() if hasattr(t, "model_dump") else t for t in out]
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


# # Test per topic node pl
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
