
from typing import List, Any, Dict, Optional, Tuple
from langgraph.graph import StateGraph, START, END
import json, copy, re
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

    # @staticmethod
    # def _find_summary_for_topic(topic_name: str, summaries_list: List[Any]) -> Optional[Any]:
    #     want = (topic_name or "").strip().lower()
    #     for s in summaries_list:
    #         nm = (QABlockGenerationAgent._get_topic_name(s) or "").strip().lower()
    #         if nm == want:
    #             return s
    #     return None

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
            deep_dive_nodes=deep_dive_nodes_json,
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

    # @staticmethod
    # async def qablock_generator(state: AgentInternalState) -> AgentInternalState:
    #     if state.interview_topics is None or not getattr(state.interview_topics, "interview_topics", None):
    #         raise ValueError("No interview topics to summarize.")
    #     if state.discussion_summary_per_topic is None:
    #         raise ValueError("discussion_summary_per_topic is required.")
    #     if state.nodes is None or not getattr(state.nodes, "topics_with_nodes", None):
    #         raise ValueError("nodes (topics_with_nodes) are required before QA block generation.")

    #     # normalize summaries_list
    #     raw = state.discussion_summary_per_topic
    #     if hasattr(raw, "discussion_topics"):
    #         summaries_list = list(raw.discussion_topics)
    #     elif isinstance(raw, (list, tuple)):
    #         summaries_list = list(raw)
    #     elif isinstance(raw, dict) and "discussion_topics" in raw:
    #         summaries_list = list(raw["discussion_topics"])
    #     else:
    #         raise ValueError("discussion_summary_per_topic has no 'discussion_topics' field")

    #     final_sets: List[Dict[str, Any]] = []
    #     accumulated_errs: List[str] = []

    #     for topic_entry in state.nodes.topics_with_nodes:
    #         topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else dict(topic_entry)
    #         topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry)

    #         summary_obj = QABlockGenerationAgent._find_summary_for_topic(topic_name, summaries_list)
    #         if summary_obj is None:
    #             msg = f"[QABlocks] No discussion summary found for topic: {topic_name}; skipping."
    #             print(msg)
    #             accumulated_errs.append(msg)
    #             continue

    #         # collect deep-dive nodes
    #         deep_dive_nodes: List[dict] = []
    #         for node in (topic_dict.get("nodes") or []):
    #             qtype = str(node.get("question_type", "")).strip().lower()
    #             if qtype == "deep dive":
    #                 deep_dive_nodes.append(node)

    #         if not deep_dive_nodes:
    #             msg = f"[QABlocks] No deep-dive nodes for topic: {topic_name}; skipping."
    #             print(msg)
    #             accumulated_errs.append(msg)
    #             continue

    #         # JSON strings for the LLM
    #         summary_json = json.dumps(
    #             summary_obj.model_dump() if hasattr(summary_obj, "model_dump") else summary_obj
    #         )
    #         deep_dive_nodes_json = json.dumps(copy.deepcopy(deep_dive_nodes))

    #         # >>> generate for THIS topic (moved inside the loop) <<<
    #         one_set, err = await QABlockGenerationAgent._gen_for_topic(
    #             topic_name=topic_name,
    #             discussion_summary_json=summary_json,
    #             deep_dive_nodes_json=deep_dive_nodes_json,
    #             qa_error=getattr(state, "qa_error", "") or ""
    #         )

    #         qa_blocks = one_set.get("qa_blocks", [])
    #         if qa_blocks:
    #             final_sets.append(one_set)
    #         else:
    #             accumulated_errs.append(f"[{topic_name}] generated 0 blocks — skipping topic")

    #         if err:
    #             accumulated_errs.append(f"[{topic_name}] {err}")

    #     # ---- after the loop, finalize state ----
    #     if final_sets:
    #         state.qa_blocks = QASetsSchema(qa_sets=final_sets)
    #     else:
    #         state.qa_blocks = None
    #         accumulated_errs.append("[QABlocks] No topics produced QA blocks; ending without validation.")

    #     if accumulated_errs:
    #         state.qa_error = (state.qa_error or "") + ("\n" if state.qa_error else "") + "\n".join(accumulated_errs)

    #     return state
    @staticmethod
    def _can(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return s

    @staticmethod
    def _make_fallback_summary(topic_name: str, topic_dict: Dict[str, Any], summaries_list: List[Any]) -> Dict[str, Any]:
        # Try to synthesize a minimal summary from topic nodes
        node_hints = []
        for n in (topic_dict.get("nodes") or [])[:5]:
            hint = n.get("title") or n.get("name") or n.get("question") or ""
            if isinstance(hint, str) and hint.strip():
                node_hints.append(hint.strip())

        return {
            "topic": topic_name,
            "summary": ("; ".join(node_hints) or "No prior discussion summary available for this topic."),
            "evidence": {"topic_node_hints": node_hints}
        }

    @staticmethod
    def _extract_topic_name_from_summary(summary_obj: Any) -> str:
        d = QABlockGenerationAgent._as_dict(summary_obj)
        for k in ("topic", "name", "title", "label", "discussion_topic", "heading"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "Unknown"
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

        # Build quick indexes (for optional pairing, not for gating)
        summaries_by_can = {}
        for s in summaries_list:
            nm = QABlockGenerationAgent._extract_topic_name_from_summary(s)
            summaries_by_can[QABlockGenerationAgent._can(nm)] = s

        DEEP_DIVE_ALIASES = {"deep dive", "deep_dive", "deep-dive", "probe", "follow up", "follow-up"}

        covered = set()  # canonical topic names already generated

        # -------- Pass A: generate for every topic node group --------
        for topic_entry in state.nodes.topics_with_nodes:
            topic_dict = topic_entry.model_dump() if hasattr(topic_entry, "model_dump") else dict(topic_entry)
            topic_name = topic_dict.get("topic") or QABlockGenerationAgent._get_topic_name(topic_entry) or "Unknown"
            ckey = QABlockGenerationAgent._can(topic_name)

            # Optional pairing (not required). If not found, make a fallback summary.
            summary_obj = summaries_by_can.get(ckey)
            if summary_obj is None:
                accumulated_errs.append(f"[QABlocks] No summary for '{topic_name}'; using fallback summary.")
                summary_obj = QABlockGenerationAgent._make_fallback_summary(topic_name, topic_dict, summaries_list)

            # Collect deep-dive nodes (fallback to a small subset if none)
            deep_dive_nodes: List[dict] = []
            for node in (topic_dict.get("nodes") or []):
                qtype = str(node.get("question_type", "")).strip().lower()
                if qtype in DEEP_DIVE_ALIASES:
                    deep_dive_nodes.append(node)
            if not deep_dive_nodes:
                nodes_all = topic_dict.get("nodes") or []
                deep_dive_nodes = nodes_all[:3]  # safe small subset
                if not deep_dive_nodes and nodes_all:
                    deep_dive_nodes = nodes_all  # if very small topic, just take what's there
                if not deep_dive_nodes:
                    accumulated_errs.append(f"[QABlocks] Topic '{topic_name}' has no nodes; proceeding with empty nodes.")

            # Prepare JSONs and invoke
            summary_json = json.dumps(summary_obj.model_dump() if hasattr(summary_obj, "model_dump") else summary_obj)
            deep_dive_nodes_json = json.dumps(copy.deepcopy(deep_dive_nodes))

            one_set, err = await QABlockGenerationAgent._gen_for_topic(
                topic_name=topic_name,
                discussion_summary_json=summary_json,
                deep_dive_nodes_json=deep_dive_nodes_json,
                qa_error=getattr(state, "qa_error", "") or ""
            )

            final_sets.append(one_set)  # always keep, even if qa_blocks is empty
            if err:
                accumulated_errs.append(f"[{topic_name}] {err}")
            elif not (one_set.get("qa_blocks") or []):
                accumulated_errs.append(f"[{topic_name}] generated 0 blocks.")

            covered.add(ckey)

        # -------- Pass B: generate for summaries that didn't appear in Pass A --------
        for s in summaries_list:
            s_name = QABlockGenerationAgent._extract_topic_name_from_summary(s) or "Unknown"
            ckey = QABlockGenerationAgent._can(s_name)
            if ckey in covered:
                continue  # already generated in Pass A

            # No nodes associated; proceed with empty or generic nodes
            summary_json = json.dumps(s.model_dump() if hasattr(s, "model_dump") else s)
            deep_dive_nodes_json = json.dumps([])

            one_set, err = await QABlockGenerationAgent._gen_for_topic(
                topic_name=s_name,
                discussion_summary_json=summary_json,
                deep_dive_nodes_json=deep_dive_nodes_json,
                qa_error=getattr(state, "qa_error", "") or ""
            )

            final_sets.append(one_set)
            if err:
                accumulated_errs.append(f"[{s_name}] {err}")
            elif not (one_set.get("qa_blocks") or []):
                accumulated_errs.append(f"[{s_name}] generated 0 blocks.")
            covered.add(ckey)

        # ---- finalize ----
        if final_sets:
            state.qa_blocks = QASetsSchema(qa_sets=final_sets)
        else:
            state.qa_blocks = None
            accumulated_errs.append("[QABlocks] No topics produced QA blocks; ending without validation.")

        if accumulated_errs:
            state.qa_error = (state.qa_error or "") + ("\n" if state.qa_error else "") + "\n".join(accumulated_errs)

        return state


    # --- should_regenerate: schema-only, but END if there's nothing to validate ---
    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:

        # Strict container validation only (as you asked)
        try:
            QASetsSchema.model_validate(
                state.qa_blocks.model_dump() if hasattr(state.qa_blocks, "model_dump") else state.qa_blocks
            )
        except ValidationError as ve:
            state.qa_error += f"The previous generated o/p did not follow the given schema as it got following errors:\n[QABlockGen ValidationError]\n {ve}"
            return True  # try again
        return False  # valid -> END

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
