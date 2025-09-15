
# from typing import List, Any, Dict, Optional
# from langgraph.graph import StateGraph, START, END
# import json, copy
# from langchain_core.messages import SystemMessage
# from pydantic import ValidationError
# from src.mongo_tools import get_mongo_tools
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import QASetsSchema
# from ..prompt.qa_agent_prompt import QA_BLOCK_AGENT_PROMPT
# from ..model_handling import llm_qa


# class QABlockGenerationAgent:
#     llm_qa = llm_qa
#     MONGO_TOOLS = get_mongo_tools(llm=llm_qa)
#     llm_qa_with_tools = llm_qa.bind_tools(MONGO_TOOLS) 

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
#     def _find_summary_for_topic(topic_name: str, summaries_list: List[Any]) -> Optional[Any]:
#         for s in summaries_list:
#             nm = QABlockGenerationAgent._get_topic_name(s)
#             if nm == topic_name:
#                 return s
#         return None

#     @staticmethod
#     async def _gen_for_topic(
#         topic_name: str,
#         discussion_summary_json: str,
#         deep_dive_nodes_json: str,
#         qa_error: str = ""
#     ) -> Dict[str, Any]:
#         """
#         Returns exactly ONE qa_set dict for the topic:
#         { "topic": "<topic_name>", "qa_blocks": [ ... ] }
#         """
#         sys = QA_BLOCK_AGENT_PROMPT.format(
#             discussion_summary=discussion_summary_json,
#             node=deep_dive_nodes_json,
#             qa_error=qa_error or ""
#         )

#         schema = await QABlockGenerationAgent.llm_qa_with_tools \
#             .with_structured_output(QASetsSchema, method="function_calling") \
#             .ainvoke([SystemMessage(content=sys)])

#         obj = schema.model_dump() if hasattr(schema, "model_dump") else schema
#         sets = obj.get("qa_sets", []) or []

#         # Enforce single-set-per-topic contract
#         if not sets:
#             return {"topic": topic_name, "qa_blocks": []}
#         one = sets[0]
#         one["topic"] = topic_name  # force consistency
#         # ensure qa_blocks exists
#         one["qa_blocks"] = one.get("qa_blocks", []) or []
#         return one

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

#         # Normalize summaries list
#         try:
#             summaries_list = list(state.discussion_summary_per_topic.discussion_topics)
#         except Exception:
#             summaries_list = list(state.discussion_summary_per_topic)

#         final_sets: List[Dict[str, Any]] = []

#         # Iterate topics ONCE; collect all Deep Dives for that topic → single call
#         for topic_entry in state.nodes.topics_with_nodes:
#             # robust Pydantic/dict handling
#             topic_dict = topic_entry.model_dump()
#             topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry)

#             summary_obj = QABlockGenerationAgent._find_summary_for_topic(topic_name, summaries_list)
#             if summary_obj is None:
#                 print(f"[QABlocks] No discussion summary found for topic: {topic_name}; skipping.")
#                 continue

#             # Collect Deep Dive nodes for this topic
#             deep_dive_nodes = []
#             for node in (topic_dict.get("nodes") or []):
#                 qtype = str(node.get("question_type", "")).strip()
#                 if qtype == "Deep Dive":
#                     deep_dive_nodes.append(node)

#             # If no deep-dive nodes, still emit an empty set for consistency
#             if not deep_dive_nodes:
#                 final_sets.append({"topic": topic_name, "qa_blocks": []})
#                 continue

#             # Prepare JSON strings
#             summary_json = json.dumps(copy.deepcopy(summary_obj.model_dump()))
#             deep_dive_nodes_json = json.dumps(copy.deepcopy(deep_dive_nodes))

#             # Single generation per topic
#             one_set = await QABlockGenerationAgent._gen_for_topic(
#                 topic_name=topic_name,
#                 discussion_summary_json=summary_json,
#                 deep_dive_nodes_json=deep_dive_nodes_json,
#                 qa_error=getattr(state, "qa_error", "") or ""
#             )
#             final_sets.append(one_set)

#         # Wrap as a single QASetsSchema with one item per topic
#         state.qa_blocks = QASetsSchema(qa_sets=final_sets)
#         return state

#     @staticmethod
#     async def should_regenerate(state: AgentInternalState) -> bool:
#         """
#         Return True if we need to regenerate (schema invalid), else False.
#         Validates the container (QASetsSchema).
#         """

#         # Validate QASetsSchema (container)
#         try:
#             # accept either a Pydantic instance or a plain dictionary
#             QASetsSchema.model_validate(
#                 state.qa_blocks.model_dump()
#             )
#             return False
#         except ValidationError as ve:
#             print("[QABlockGen][ValidationError] Container QASetsSchema invalid")
#             print(str(ve))
#             state.qa_error += "The previous generated o/p did not follow the given schema as it got following errors:\n" + \
#                                 "\n[QABlockGen ValidationError]\n" + str(ve) + "\n"
#             return True

#     @staticmethod
#     def get_graph(checkpointer=None):
#         gb = StateGraph(state_schema=AgentInternalState)
#         gb.add_node("qablock_generator", QABlockGenerationAgent.qablock_generator)
#         gb.add_edge(START, "qablock_generator")
#         gb.add_conditional_edges(
#             "qablock_generator",
#             QABlockGenerationAgent.should_regenerate,  # returns True/False
#             { True: "qablock_generator", False: END }
#         )
#         return gb.compile(checkpointer=checkpointer, name="QA Block Generation Agent")


# if __name__ == "__main__":
#     graph = QABlockGenerationAgent.get_graph()
#     g = graph.get_graph().draw_ascii()
#     print(g)


from typing import List, Any, Dict, Optional, Tuple
from langgraph.graph import StateGraph, START, END
import json, copy
from langchain_core.messages import SystemMessage
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import QASetsSchema
from ..prompt.qa_agent_prompt import QA_BLOCK_AGENT_PROMPT
from ..model_handling import llm_qa
from src.utils import load_config

config = load_config("config.yaml")
thread_id = config["configurable"]["thread_id"] 

class QABlockGenerationAgent:
    llm_qa = llm_qa
    MONGO_TOOLS = get_mongo_tools(llm=llm_qa)
    llm_qa_with_tools = llm_qa.bind_tools(MONGO_TOOLS)

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
    ) -> Tuple[Dict[str, Any], str]:
        """
        Returns (one_qa_set_dict, err_msg).
        On parse/validation error, returns ({ "topic": topic_name, "qa_blocks": [] }, error_string).
        """
        sys = QA_BLOCK_AGENT_PROMPT.format(
            discussion_summary=discussion_summary_json,
            node=deep_dive_nodes_json,
            qa_error=qa_error or "",
            thread_id=thread_id
        )

        try:
            schema = await QABlockGenerationAgent.llm_qa_with_tools \
                .with_structured_output(QASetsSchema, method="function_calling") \
                .ainvoke([SystemMessage(content=sys)])

            obj = schema.model_dump() if hasattr(schema, "model_dump") else schema
            sets = obj.get("qa_sets", []) or []
            if not sets:
                return {"topic": topic_name, "qa_blocks": []}, "No qa_sets produced."

            one = sets[0]
            one["topic"] = topic_name
            one["qa_blocks"] = one.get("qa_blocks", []) or []
            return one, ""
        except (ValidationError, OutputParserException) as e:
            return {"topic": topic_name, "qa_blocks": []}, f"Parser/Schema error: {e}"

    @staticmethod
    async def qablock_generator(state: AgentInternalState) -> AgentInternalState:
        if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
            raise ValueError("No interview topics to summarize.")
        if state.discussion_summary_per_topic is None:
            raise ValueError("discussion_summary_per_topic is required.")
        if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
            raise ValueError("nodes (topics_with_nodes) are required before QA block generation.")

        # normalize summaries_list
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

        for topic_entry in state.nodes.topics_with_nodes:
            topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else dict(topic_entry)
            topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry)

            summary_obj = QABlockGenerationAgent._find_summary_for_topic(topic_name, summaries_list)
            if summary_obj is None:
                print(f"[QABlocks] No discussion summary found for topic: {topic_name}; skipping.")
                final_sets.append({"topic": topic_name, "qa_blocks": []})
                continue

            # collect deep-dive nodes
            deep_dive_nodes: List[dict] = []
            for node in (topic_dict.get("nodes") or []):
                qtype = str(node.get("question_type", "")).strip().lower()
                if qtype == "deep dive":
                    deep_dive_nodes.append(node)

            if not deep_dive_nodes:
                final_sets.append({"topic": topic_name, "qa_blocks": []})
                continue

            # JSON strings for the LLM
            summary_json = json.dumps(
                summary_obj.model_dump() if hasattr(summary_obj, "model_dump") else summary_obj
            )
            deep_dive_nodes_json = json.dumps(copy.deepcopy(deep_dive_nodes))

            # generate for this topic
            one_set, err = await QABlockGenerationAgent._gen_for_topic(
                topic_name=topic_name,
                discussion_summary_json=summary_json,
                deep_dive_nodes_json=deep_dive_nodes_json,
                qa_error=getattr(state, "qa_error", "") or ""
            )
            final_sets.append(one_set)
            if err:
                accumulated_errs.append(f"[{topic_name}] {err}")

        state.qa_blocks = QASetsSchema(qa_sets=final_sets)
        print(state.qa_blocks)
        if accumulated_errs:
            state.qa_error = (state.qa_error or "") + ("\n" if state.qa_error else "") + "\n".join(accumulated_errs)
        return state

    # @staticmethod
    # async def should_regenerate(state: AgentInternalState) -> bool:
    #     """
    #     Return True to REGENERATE (invalid), False to END (valid).
    #     Validates container + exact EMH/EMH/MH combo per QA block and 5 examples per QA.
    #     """
    #     # 0) must exist
    #     if not getattr(state, "qa_blocks", None):
    #         state.qa_error = (state.qa_error or "") + "\n[QABlockGen] qa_blocks is missing"
    #         return True

    #     # 1) container validation
    #     try:
    #         payload = state.qa_blocks.model_dump() if hasattr(state.qa_blocks, "model_dump") else state.qa_blocks
    #         QASetsSchema.model_validate(payload)
    #     except ValidationError as ve:
    #         state.qa_error = (state.qa_error or "") + f"\n[QABlockGen ValidationError] {ve}"
    #         return True

    #     # 2) exact combo guard per block
    #     def to_dict_list(items: Any) -> List[Dict[str, Any]]:
    #         # normalize list of Pydantic models / dicts to list[dict]
    #         lst = items or []
    #         return [
    #             (i.model_dump() if hasattr(i, "model_dump") else dict(i))
    #             for i in lst
    #         ]

    #     def combos_ok(qa_items_raw: Any) -> bool:
    #         qa_items = to_dict_list(qa_items_raw)
    #         if len(qa_items) != 8:
    #             return False

    #         counts = {
    #             ("First Question", "Easy"): 0,
    #             ("First Question", "Medium"): 0,
    #             ("First Question", "Hard"): 0,
    #             ("New Question", "Easy"): 0,
    #             ("New Question", "Medium"): 0,
    #             ("New Question", "Hard"): 0,
    #             ("Counter Question", "Medium"): 0,
    #             ("Counter Question", "Hard"): 0,
    #         }
    #         for it in qa_items:
    #             qt = it.get("q_type")
    #             qd = it.get("q_difficulty")
    #             key = (qt, qd)
    #             if key not in counts:
    #                 # any disallowed pair (e.g., Counter+Easy) → fail
    #                 return False

    #             # exactly 5 example questions, non-empty strings
    #             ex = it.get("example_questions") or []
    #             if not (isinstance(ex, list) and len(ex) == 5 and all(isinstance(s, str) and s.strip() for s in ex)):
    #                 return False

    #             counts[key] += 1

    #         # require exactly one of each combo
    #         return all(v == 1 for v in counts.values())

    #     problems = []
    #     for s in (state.qa_blocks.qa_sets or []):
    #         blocks = s.qa_blocks if hasattr(s, "qa_blocks") else (s.get("qa_blocks") or [])
    #         for b in blocks:
    #             b_dict = b.model_dump() if hasattr(b, "model_dump") else dict(b)
    #             items = b_dict.get("qa_items")
    #             if not combos_ok(items):
    #                 problems.append(f"Bad combo/count in block {b_dict.get('block_id','?')}")

    #     if problems:
    #         state.qa_error = (state.qa_error or "") + ("\n" if state.qa_error else "") + "\n".join(problems)
    #         return True

    #     return False

    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        # --- must exist ---
        if not getattr(state, "qa_blocks", None):
            state.qa_error = (state.qa_error or "") + "\n[QABlockGen] qa_blocks is missing"
            return True

        # --- container validation ---
        try:
            payload = (
                state.qa_blocks.model_dump()
                if hasattr(state.qa_blocks, "model_dump")
                else state.qa_blocks
            )
            # Use parse_obj instead of model_validate for stricter schema enforcement
            QASetsSchema.model_validate_json(json.dumps(payload))
        except ValidationError as ve:
            state.qa_error = (state.qa_error or "") + f"\n[QABlockGen ValidationError] {ve}"
            return True

        return False


    @staticmethod
    def get_graph(checkpointer=None):
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
    g = graph.get_graph().draw_ascii()
    print(g)
