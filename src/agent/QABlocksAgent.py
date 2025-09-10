
# from typing import List, Any, Dict, Optional
# from langgraph.graph import StateGraph, START, END
# import json, copy
# from langchain_core.messages import SystemMessage
# from pydantic import ValidationError
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import QASetsSchema
# from ..prompt.qa_agent_prompt import QA_BLOCK_AGENT_PROMPT
# from ..model_handling import llm_n


# class QABlockGenerationAgent:
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
#         d = QABlockGenerationAgent._as_dict(obj)
#         for k in ("topic", "name", "title", "label"):
#             v = d.get(k)
#             if isinstance(v, str) and v.strip():
#                 return v.strip()
#         return "Unknown"

#     @staticmethod
#     def _find_summary_for_topic(
#         topic_name: str,
#         summaries_list: List[Any]
#     ) -> Optional[Any]:
#         # summaries are DiscussionTopic objects or dicts with .topic / ["topic"]
#         for s in summaries_list:
#             nm = QABlockGenerationAgent._get_topic_name(s)
#             if nm == topic_name:
#                 return s
#         return None

#     # -----------------------------
#     # Single generation
#     # -----------------------------
#     @staticmethod
#     async def _gen_once(
#         discussion_summary_json: str,
#         node_json: str,
#         qa_error: str = ""
#     ) -> QASetsSchema:
#         sys = QA_BLOCK_AGENT_PROMPT.format(
#             discussion_summary=discussion_summary_json,
#             node=node_json,
#             qa_error=qa_error or ""
#         )
#         return await QABlockGenerationAgent.llm_n \
#             .with_structured_output(QASetsSchema, method="function_calling") \
#             .ainvoke([SystemMessage(content=sys)])

#     # -----------------------------
#     # Main generator
#     # -----------------------------
#     @staticmethod
#     async def qablock_generator(state: AgentInternalState) -> AgentInternalState:

#         if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
#             raise ValueError("No interview topics to summarize.")
#         if state.discussion_summary_per_topic is None:
#             raise ValueError("discussion_summary_per_topic is required.")
#         if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
#             raise ValueError("nodes (topics_with_nodes) are required before QA block generation.")

#         # Normalize lists
#         try:
#             summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
#         except Exception:
#             summaries_list = list(state.discussion_summary_per_topic)

#         # Accumulator: build a single QASetsSchema by concatenation
#         aggregated_sets = []

#         # Iterate each topic’s nodes, emit QA blocks for Deep Dive nodes
#         for topic_entry in state.nodes.topics_with_nodes:
#             # topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else topic_entry
#             topic_dict = json.loads(topic_entry.model_dump_json())
#             topic_name = topic_dict.get("topic")

#             # Find matching discussion summary for this topic
#             summary_obj = QABlockGenerationAgent._find_summary_for_topic(topic_name, summaries_list)
#             if summary_obj is None:
#                 # If no summary found, skip safely or raise based on your preference
#                 print(f"[QABlocks] No discussion summary found for topic: {topic_name}; skipping.")
#                 continue

#             # Prepare JSON strings for the prompt
#             summary_json = copy.deepcopy(summary_obj.model_dump())

#             nodes_list = topic_dict.get("nodes", [])
#             for node_current_topic in nodes_list:
#                 node_dict = node_current_topic
#                 qtype = str(node_dict.get("question_type", "")).strip()
#                 if qtype != "Deep Dive":
#                     continue

#                 node_json = json.dumps(copy.deepcopy(node_dict))
#                 # Generate per-node QA blocks; returns a QASetsSchema
#                 result_schema = await QABlockGenerationAgent._gen_once(
#                     discussion_summary_json=summary_json,
#                     node_json=node_json,
#                     qa_error=getattr(state, "qa_error", "") or ""
#                 )


#                 result_obj = json.loads(result_schema.model_dump_json())
#                 sets = result_obj.get("qa_sets", [])
#                 if sets:
#                     aggregated_sets.extend(sets)

#         # Build a single QASetsSchema
#         final_schema = QASetsSchema(qa_sets=aggregated_sets) if aggregated_sets else QASetsSchema(qa_sets=[])

#         # Assign to state as a single model, not a list
#         state.qa_blocks = final_schema
#         return state

#     @staticmethod
#     def get_graph(checkpointer=None):
#         gb = StateGraph(state_schema=AgentInternalState)
#         gb.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
#         gb.add_edge(START, "qablock_generator")
#         # Optionally add a validator + retry loop later
#         return gb.compile(checkpointer=checkpointer, name="QA Block Generation Agent")


# if __name__ == "__main__":
#     graph = QABlockGenerationAgent.get_graph()
#     g = graph.get_graph().draw_ascii()
#     print(g)


from typing import List, Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
import json, copy
from langchain_core.messages import SystemMessage
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import QASetsSchema
from ..prompt.qa_agent_prompt import QA_BLOCK_AGENT_PROMPT
from ..model_handling import llm_n


class QABlockGenerationAgent:
    llm_n = llm_n

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
    def _find_summary_for_topic(topic_name: str, summaries_list: List[Any]) -> Optional[Any]:
        for s in summaries_list:
            nm = QABlockGenerationAgent._get_topic_name(s)
            if nm == topic_name:
                return s
        return None

    @staticmethod
    async def _gen_for_topic(
        topic_name: str,
        discussion_summary_json: str,
        deep_dive_nodes_json: str,
        qa_error: str = ""
    ) -> Dict[str, Any]:
        """
        Returns exactly ONE qa_set dict for the topic:
        { "topic": "<topic_name>", "qa_blocks": [ ... ] }
        """
        sys = QA_BLOCK_AGENT_PROMPT.format(
            topic=topic_name,
            discussion_summary=discussion_summary_json,
            node=deep_dive_nodes_json,
            qa_error=qa_error or ""
        )
        schema = await QABlockGenerationAgent.llm_n \
            .with_structured_output(QASetsSchema, method="function_calling") \
            .ainvoke([SystemMessage(content=sys)])

        obj = schema.model_dump() if hasattr(schema, "model_dump") else schema
        sets = obj.get("qa_sets", []) or []

        # Enforce single-set-per-topic contract
        if not sets:
            return {"topic": topic_name, "qa_blocks": []}
        one = sets[0]
        one["topic"] = topic_name  # force consistency
        # ensure qa_blocks exists
        one["qa_blocks"] = one.get("qa_blocks", []) or []
        return one

    # -----------------------------
    # Main generator
    # -----------------------------
    @staticmethod
    async def qablock_generator(state: AgentInternalState) -> AgentInternalState:
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")
        if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
            raise ValueError("nodes (topics_with_nodes) are required before QA block generation.")

        # Normalize summaries list
        try:
            summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
        except Exception:
            summaries_list = list(state.discussion_summary_per_topic)

        final_sets: List[Dict[str, Any]] = []

        # Iterate topics ONCE; collect all Deep Dives for that topic → single call
        for topic_entry in state.nodes.topics_with_nodes:
            # robust Pydantic/dict handling
            # topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else topic_entry
            topic_dict = topic_entry.model_dump()
            topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry)

            summary_obj = QABlockGenerationAgent._find_summary_for_topic(topic_name, summaries_list)
            if summary_obj is None:
                print(f"[QABlocks] No discussion summary found for topic: {topic_name}; skipping.")
                continue

            # Collect Deep Dive nodes for this topic
            deep_dive_nodes = []
            for node in (topic_dict.get("nodes") or []):
                qtype = str(node.get("question_type", "")).strip()
                if qtype == "Deep Dive":
                    deep_dive_nodes.append(node)

            # If no deep-dive nodes, still emit an empty set for consistency
            if not deep_dive_nodes:
                final_sets.append({"topic": topic_name, "qa_blocks": []})
                continue

            # Prepare JSON strings
            # summary_json = json.dumps(copy.deepcopy(summary_obj.model_dump() if hasattr(summary_obj, "model_dump") else summary_obj))
            summary_json = json.dumps(copy.deepcopy(summary_obj.model_dump()))
            deep_dive_nodes_json = json.dumps(copy.deepcopy(deep_dive_nodes))

            # Single generation per topic
            one_set = await QABlockGenerationAgent._gen_for_topic(
                topic_name=topic_name,
                discussion_summary_json=summary_json,
                deep_dive_nodes_json=deep_dive_nodes_json,
                qa_error=getattr(state, "qa_error", "") or ""
            )
            final_sets.append(one_set)

        # Wrap as a single QASetsSchema with one item per topic
        state.qa_blocks = QASetsSchema(qa_sets=final_sets)
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
        gb.add_edge(START, "qablock_generator")
        return gb.compile(checkpointer=checkpointer, name="QA Block Generation Agent")


if __name__ == "__main__":
    graph = QABlockGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
