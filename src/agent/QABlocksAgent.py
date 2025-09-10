
# from typing import List, Any, Dict
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
#     def _to_json_one(x: Any) -> str:
#         # deep copy -> dict -> json (avoid mutating upstream structures)
#         if hasattr(x, "model_dump"):
#             return json.dumps(copy.deepcopy(x.model_dump()))
#         if hasattr(x, "dict"):
#             return json.dumps(copy.deepcopy(x.dict()))
#         return json.dumps(copy.deepcopy(x))

#     @staticmethod
#     def _get_total_questions(topic_obj: Any, dspt_obj: Any) -> int:

#         for src in (topic_obj, dspt_obj):
#             d = QABlockGenerationAgent._as_dict(src)
#             tq = d.get("total_questions")
#             if isinstance(tq, int) and tq >= 2:
#                 return tq
#         raise ValueError("total_questions must be >= 2 for each topic")

#     @staticmethod
#     async def _gen_once(discussion_summary: str, node, T, nodes_error="") -> QASetsSchema:
#         sys = QA_BLOCK_AGENT_PROMPT.format(
#             discussion_summary=discussion_summary,
#             node=node,
#             nodes_error=nodes_error
#         )
#         return await QABlockGenerationAgent.llm_n \
#             .with_structured_output(QASetsSchema, method="function_calling") \
#             .ainvoke([SystemMessage(content=sys)])

#     # @staticmethod
#     # async def should_regenerate(state: AgentInternalState) -> bool:
#     #     """
#     #     Return True if we need to regenerate (schema invalid), else False.
#     #     Validates the container (NodesSchema) and each TopicWithNodesSchema item inside it.
#     #     """
#     #     # Nothing produced yet? -> regenerate
#     #     if getattr(state, "nodes", None) is None:
#     #         return True

#     #     # Validate NodesSchema (container)
#     #     try:
#     #         # accept either a Pydantic instance or a plain dictionary
#     #         NodesSchema.model_validate(
#     #             state.nodes.model_dump() if hasattr(state.nodes, "model_dump") else state.nodes
#     #         )
#     #     except ValidationError as ve:
#     #         print("[NodesGen][ValidationError] Container NodesSchema invalid")
#     #         print(str(ve))
#     #         state.nodes_error += "The previous generated o/p did not follow the given schema as it got following errors:\n" + (getattr(state, "nodes_error", "") or "") + \
#     #                             "\n[NodesSchema ValidationError]\n" + str(ve) + "\n"
#     #         return True

#     #     # Validate each topic payload
#     #     try:
#     #         topics_payload = (
#     #             state.nodes.topics_with_nodes
#     #             if hasattr(state.nodes, "topics_with_nodes")
#     #             else state.nodes.get("topics_with_nodes", [])
#     #         )
#     #     except Exception as e:
#     #         print("[NodesGen][ValidationError] Could not read topics_with_nodes:", e)
#     #         state.nodes_error += (getattr(state, "nodes_error", "") or "") + \
#     #                             "\n[NodesSchema Payload Error]\n" + str(e) + "\n"
#     #         return True

#     #     any_invalid = False
#     #     for idx, item in enumerate(topics_payload):
#     #         try:
#     #             # item can be pydantic model or dictionary
#     #                 QASetsSchema.model_validate(
#     #                 item.model_dump() if hasattr(item, "model_dump") else item
#     #             )
#     #         except ValidationError as ve:
#     #             any_invalid = True
#     #             print(f"[NodesGen][ValidationError] TopicWithNodesSchema invalid at index {idx}")
#     #             print(str(ve))
#     #             state.nodes_error += (getattr(state, "nodes_error", "") or "") + \
#     #                                 f"\n[TopicWithNodesSchema ValidationError idx={idx}]\n" + str(ve) + "\n"

#     #     return any_invalid

#     # -----------------------------
#     # Main Node generation node
#     # -----------------------------
#     @staticmethod
#     async def qablock_generator(state: AgentInternalState) -> AgentInternalState:

#         if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
#             raise ValueError("No interview topics to summarize.")
#         if state.discussion_summary_per_topic is None:
#             raise ValueError("discussion_summary_per_topic is required.")

#         try:
#             summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
#         except Exception:
#             summaries_list = list(state.discussion_summary_per_topic)

#         resp:QASetsSchema = []
#         for topic in state.nodes.topics_with_nodes:
#             for node_current_topic in topic.nodes:
#                 if str(node_current_topic.question_type) == "Deep Dive":
#                     print(f"-------{node_current_topic.model_dump_json()}--------")
#                     resp += [await QABlockGenerationAgent._gen_once(summaries_list, node_current_topic.model_dump_json(), state.qa_error)]
#         state.qa_blocks = resp
#         return state

#     @staticmethod
#     def get_graph(checkpointer=None):
#         gb = StateGraph(state_schema=AgentInternalState)
#         gb.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
#         gb.add_edge(START, "qablock_generator")
#         # gb.add_conditional_edges(
#         #     "qablock_generator",
#         #     QABlockGenerationAgent.should_regenerate,  # returns True/False
#         #     { True: "qablock_generator", False: END }
#         # )
#         return gb.compile(checkpointer=checkpointer, name="QA Block Generation Agent")

# if __name__ == "__main__":
#     graph = QABlockGenerationAgent.get_graph()
#     g = graph.get_graph().draw_ascii()
#     print(g)


from typing import List, Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
import json, copy
from langchain_core.messages import SystemMessage
from pydantic import ValidationError
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import QASetsSchema, QASet, QABlock  # make sure these are exported
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
    def _to_json_one(x: Any) -> str:
        # deep copy -> dict -> json (avoid mutating upstream structures)
        if hasattr(x, "model_dump"):
            return json.dumps(copy.deepcopy(x.model_dump()))
        if hasattr(x, "dict"):
            return json.dumps(copy.deepcopy(x.dict()))
        return json.dumps(copy.deepcopy(x))


    @staticmethod
    def _find_summary_for_topic(
        topic_name: str,
        summaries_list: List[Any]
    ) -> Optional[Any]:
        # summaries are DiscussionTopic objects or dicts with .topic / ["topic"]
        for s in summaries_list:
            nm = QABlockGenerationAgent._get_topic_name(s)
            if nm == topic_name:
                return s
        return None

    # -----------------------------
    # Single generation
    # -----------------------------
    @staticmethod
    async def _gen_once(
        discussion_summary_json: str,
        node_json: str,
        qa_error: str = ""
    ) -> QASetsSchema:
        sys = QA_BLOCK_AGENT_PROMPT.format(
            discussion_summary=discussion_summary_json,
            node=node_json,
            qa_error=qa_error or ""
        )
        return await QABlockGenerationAgent.llm_n \
            .with_structured_output(QASetsSchema, method="function_calling") \
            .ainvoke([SystemMessage(content=sys)])

    # -----------------------------
    # Main generator
    # -----------------------------
    @staticmethod
    async def qablock_generator(state: AgentInternalState) -> AgentInternalState:
        # Guards
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")
        if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
            raise ValueError("nodes (topics_with_nodes) are required before QA block generation.")

        # Normalize lists
        try:
            summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
        except Exception:
            summaries_list = list(state.discussion_summary_per_topic)

        # Accumulator: build a single QASetsSchema by concatenation
        aggregated_sets = []

        # Iterate each topic’s nodes, emit QA blocks for Deep Dive nodes
        for topic_entry in state.nodes.topics_with_nodes:
            # topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else topic_entry
            topic_dict = json.loads(topic_entry.model_dump_json())
            topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry)

            # Find matching discussion summary for this topic
            summary_obj = QABlockGenerationAgent._find_summary_for_topic(topic_name, summaries_list)
            if summary_obj is None:
                # If no summary found, skip safely or raise based on your preference
                print(f"[QABlocks] No discussion summary found for topic: {topic_name}; skipping.")
                continue

            # # Determine total_questions from either the topic or its summary
            # try:
            #     T = QABlockGenerationAgent._get_total_questions(topic_entry, summary_obj)
            # except ValueError:
            #     # If not available, you can choose a default or raise. Here we raise for correctness.
            #     raise

            # Prepare JSON strings for the prompt
            summary_json = QABlockGenerationAgent._to_json_one(summary_obj)

            nodes_list = topic_dict.get("nodes", [])
            for node_current_topic in nodes_list:
                # node_dict = node_current_topic if isinstance(node_current_topic, dict) else (
                #     node_current_topic.model_dump() if hasattr(node_current_topic, "model_dump") else {}
                # )
                node_dict = node_current_topic
                qtype = str(node_dict.get("question_type", "")).strip()
                if qtype != "Deep Dive":
                    continue

                # print(f"-------DeepDive node for topic '{topic_name}': {node_dict}--------")

                node_json = json.dumps(copy.deepcopy(node_dict))
                # Generate per-node QA blocks; returns a QASetsSchema
                result_schema = await QABlockGenerationAgent._gen_once(
                    discussion_summary_json=summary_json,
                    node_json=node_json,
                    qa_error=getattr(state, "qa_error", "") or ""
                )

                # # Merge: extend aggregated qa_sets with result qa_sets
                # if hasattr(result_schema, "model_dump"):
                #     result_obj = result_schema.model_dump()
                # else:
                #     result_obj = result_schema

                result_obj = json.loads(result_schema.model_dump_json())
                sets = result_obj.get("qa_sets", [])
                if sets:
                    aggregated_sets.extend(sets)

        # Build a single QASetsSchema
        final_schema = QASetsSchema(qa_sets=aggregated_sets) if aggregated_sets else QASetsSchema(qa_sets=[])

        # Assign to state as a single model, not a list
        state.qa_blocks = final_schema
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
        gb.add_edge(START, "qablock_generator")
        # Optionally add a validator + retry loop later
        return gb.compile(checkpointer=checkpointer, name="QA Block Generation Agent")


if __name__ == "__main__":
    graph = QABlockGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
