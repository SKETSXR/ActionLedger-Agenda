# # Running with schema check and regenerate
# from typing import List, Any, Dict
# from langgraph.graph import StateGraph, START, END
# import json, copy
# from langchain_core.messages import SystemMessage
# from pydantic import ValidationError
# from src.mongo_tools import get_mongo_tools
# from ..schema.agent_schema import AgentInternalState
# from ..schema.output_schema import NodesSchema, TopicWithNodesSchema
# from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
# from ..model_handling import llm_n


# class NodesGenerationAgent:
#     llm_n = llm_n
#     MONGO_TOOLS = get_mongo_tools(llm=llm_n)
#     llm_n_with_tools = llm_n.bind_tools(MONGO_TOOLS) 

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
#             d = NodesGenerationAgent._as_dict(src)
#             tq = d.get("total_questions")
#             if isinstance(tq, int) and tq >= 2:
#                 return tq
#         raise ValueError("total_questions must be >= 2 for each topic")

#     @staticmethod
#     async def _gen_once(per_topic_summary_json: str, T, nodes_error, thread_id) -> TopicWithNodesSchema:
#         sys = NODES_AGENT_PROMPT.format(
#             per_topic_summary_json=per_topic_summary_json,
#             total_no_questions_context=T,
#             thread_id=thread_id,
#             nodes_error=nodes_error
#         )
#         return await NodesGenerationAgent.llm_n_with_tools \
#             .with_structured_output(TopicWithNodesSchema, method="function_calling") \
#             .ainvoke([SystemMessage(content=sys)])

#     @staticmethod
#     async def should_regenerate(state: AgentInternalState) -> bool:
#         """
#         Return True if we need to regenerate (schema invalid), else False.
#         Validates the container (NodesSchema) and each TopicWithNodesSchema item inside it.
#         """
#         # Nothing produced yet? -> regenerate
#         if getattr(state, "nodes", None) is None:
#             return True

#         # Validate NodesSchema (container)
#         try:
#             # accept either a Pydantic instance or a plain dictionary
#             NodesSchema.model_validate(
#                 state.nodes.model_dump() if hasattr(state.nodes, "model_dump") else state.nodes
#             )
#         except ValidationError as ve:
#             print("[NodesGen][ValidationError] Container NodesSchema invalid")
#             print(str(ve))
#             state.nodes_error += "The previous generated o/p did not follow the given schema as it got following errors:\n" + (getattr(state, "nodes_error", "") or "") + \
#                                 "\n[NodesSchema ValidationError]\n" + str(ve) + "\n"
#             return True

#         # Validate each topic payload
#         try:
#             topics_payload = (
#                 state.nodes.topics_with_nodes
#                 if hasattr(state.nodes, "topics_with_nodes")
#                 else state.nodes.get("topics_with_nodes", [])
#             )
#         except Exception as e:
#             print("[NodesGen][ValidationError] Could not read topics_with_nodes:", e)
#             state.nodes_error += "\n[NodesSchema Payload Error]\n" + str(e) + "\n"
#             return True

#         any_invalid = False
#         for idx, item in enumerate(topics_payload):
#             try:
#                 # item can be pydantic model or dictionary
#                 TopicWithNodesSchema.model_validate(
#                     item.model_dump() if hasattr(item, "model_dump") else item
#                 )
#             except ValidationError as ve:
#                 any_invalid = True
#                 print(f"[NodesGen][ValidationError] TopicWithNodesSchema invalid at index {idx}")
#                 print(str(ve))
#                 state.nodes_error += f"\n[TopicWithNodesSchema ValidationError idx={idx}]\n" + str(ve) + "\n"

#         return any_invalid

#     # -----------------------------
#     # Main Node generation node
#     # -----------------------------
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

#         # Snapshot upstream summaries (sanity: ensure we don't mutate them)
#         snapshot = json.dumps(
#             [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
#             sort_keys=True
#         )

#         out: List[TopicWithNodesSchema] = []

#         for topic_obj, dspt_obj in zip(topics_list, summaries_list):
#             T = NodesGenerationAgent._get_total_questions(topic_obj, dspt_obj)
#             per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)

#             resp = await NodesGenerationAgent._gen_once(per_topic_summary_json, T, state.id, state.nodes_error)
#             out.append(resp)

#         # Verify upstream summary wasn’t mutated
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
#         gb = StateGraph(state_schema=AgentInternalState)
#         gb.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
#         gb.add_edge(START, "nodes_generator")
#         gb.add_conditional_edges(
#             "nodes_generator",
#             NodesGenerationAgent.should_regenerate,  # returns True/False
#             { True: "nodes_generator", False: END }
#         )
#         return gb.compile(checkpointer=checkpointer, name="Nodes Generation Agent")

# if __name__ == "__main__":
#     graph = NodesGenerationAgent.get_graph()
#     g = graph.get_graph().draw_ascii()
#     print(g)


# Running with schema check and regenerate
from typing import List, Any, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
import json, copy
from pydantic import ValidationError
from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import NodesSchema, TopicWithNodesSchema
from ..prompt.nodes_agent_prompt import NODES_AGENT_PROMPT
from ..model_handling import llm_n

i = 1

# At top of file (if you added the log helpers there)
def _log_planned_tool_calls(ai_msg):
    for tc in getattr(ai_msg, "tool_calls", []) or []:
        try:
            print(f"[ToolCall] name={tc['name']} args={tc.get('args')}")
        except Exception:
            print(f"[ToolCall] {tc}")

def _log_recent_tool_results(messages):
    i = len(messages) - 1
    while i >= 0 and getattr(messages[i], "type", None) == "tool":
        tm = messages[i]
        print(f"[ToolResult] tool_call_id={getattr(tm, 'tool_call_id', None)} result={tm.content}")
        i -= 1



# ---------- Inner ReAct state for Mongo loop (per-topic) ----------
class _MongoNodesState(MessagesState):
    final_response: TopicWithNodesSchema


class NodesGenerationAgent:
    llm_n = llm_n
    # Mongo tools as a FLAT list; bind to model for tool-calling
    # Inside class TopicGenerationAgent
    # make sure you have this:
    MONGO_TOOLS = get_mongo_tools(llm=llm_n)
    _AGENT_MODEL = llm_n.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_n.with_structured_output(
        TopicWithNodesSchema, method="function_calling"
    )


    # ---------- Inner graph (agent -> tools -> agent ... -> respond) ----------
    @staticmethod
    def _agent_node(state: _MongoNodesState):
        # If we just came from ToolNode, the last messages are ToolMessages → print them.
        print("----------------Nodes Tool Call logs-----------------------------------")
        _log_recent_tool_results(state["messages"])   # optional logging

        ai = NodesGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    def _respond_node(state: _MongoNodesState):
        msgs = state["messages"]

        # Find the last ASSISTANT message that does NOT request tools
        ai_content = None
        for m in reversed(msgs):
            # LangChain AssistantMessage typically has .type == "ai"
            if getattr(m, "type", None) in ("ai", "assistant"):
                if not getattr(m, "tool_calls", None):  # completed answer
                    ai_content = m.content
                    break

        # Fallbacks: if none found, take the last assistant content even if it had tool calls,
        # else take the last message content.
        if ai_content is None:
            for m in reversed(msgs):
                if getattr(m, "type", None) in ("ai", "assistant"):
                    ai_content = m.content
                    break
        if ai_content is None:
            ai_content = msgs[-1].content

        final_obj = NodesGenerationAgent._STRUCTURED_MODEL.invoke(
            [HumanMessage(content=ai_content)]
        )
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoNodesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            _log_planned_tool_calls(last)  # optional logging
            return "continue"
        return "respond"

    _workflow = StateGraph(_MongoNodesState)
    _workflow.add_node("agent", _agent_node)
    _workflow.add_node("respond", _respond_node)
    _workflow.add_node("tools", ToolNode(MONGO_TOOLS, tags=["mongo-tools"]))
    _workflow.set_entry_point("agent")
    _workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"continue": "tools", "respond": "respond"},
    )
    _workflow.add_edge("tools", "agent")
    _workflow.add_edge("respond", END)
    _nodes_graph = _workflow.compile()

    # ----------------- helpers (unchanged) -----------------
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

    # ----------------- per-topic generation using inner graph -----------------
    @staticmethod
    async def _gen_once(per_topic_summary_json: str, T, nodes_error, thread_id) -> TopicWithNodesSchema:
        from string import Template

        class AtTemplate(Template):
            delimiter = '@'   # anything not used in your prompt samples

        tpl = AtTemplate(NODES_AGENT_PROMPT)
        content = tpl.substitute(
            per_topic_summary_json=per_topic_summary_json,
            total_no_questions_context=T,
            thread_id=thread_id,
            nodes_error=nodes_error,
        )
        sys = content

        # Drive inner graph: agent <-> tools ... -> respond (structured)
        result = await NodesGenerationAgent._nodes_graph.ainvoke(
            {"messages": [SystemMessage(content=sys)]}
        )
        return result["final_response"]  # TopicWithNodesSchema

    # ----------------------------- Main Node generation node -----------------------------
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

        # Snapshot upstream summaries to ensure we don't mutate them
        snapshot = json.dumps(
            [s.model_dump() if hasattr(s, "model_dump") else s for s in summaries_list],
            sort_keys=True
        )

        out: List[TopicWithNodesSchema] = []

        for topic_obj, dspt_obj in zip(topics_list, summaries_list):
            T = NodesGenerationAgent._get_total_questions(topic_obj, dspt_obj)
            per_topic_summary_json = NodesGenerationAgent._to_json_one(dspt_obj)

            # NOTE: argument order matches _gen_once signature
            resp = await NodesGenerationAgent._gen_once(
                per_topic_summary_json,
                T,
                state.nodes_error,
                state.id,
            )
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
    async def should_regenerate(state: AgentInternalState) -> bool:
        """
        Return True if we need to regenerate (schema invalid), else False.
        Validates the container (NodesSchema) and each TopicWithNodesSchema item inside it.
        """
        global i 
        print(f"Nodes Iteration -> {i}")
        i += 1
        if getattr(state, "nodes", None) is None:
            return True

        # Validate NodesSchema (container)
        try:
            NodesSchema.model_validate(
                state.nodes.model_dump() if hasattr(state.nodes, "model_dump") else state.nodes
            )
        except ValidationError as ve:
            print("[NodesGen][ValidationError] Container NodesSchema invalid")
            print(str(ve))
            state.nodes_error += (
                "The previous generated o/p did not follow the given schema as it got following errors:\n"
                + (getattr(state, "nodes_error", "") or "")
                + "\n[NodesSchema ValidationError]\n"
                + str(ve)
                + "\n"
            )
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
                TopicWithNodesSchema.model_validate(
                    item.model_dump() if hasattr(item, "model_dump") else item
                )
            except ValidationError as ve:
                any_invalid = True
                print(f"[NodesGen][ValidationError] TopicWithNodesSchema invalid at index {idx}")
                print(str(ve))
                state.nodes_error += (
                    f"\n[TopicWithNodesSchema ValidationError idx={idx}]\n" + str(ve) + "\n"
                )

        return any_invalid

    @staticmethod
    def get_graph(checkpointer=None):
        gb = StateGraph(state_schema=AgentInternalState)
        gb.add_node("nodes_generator", NodesGenerationAgent.nodes_generator)
        gb.add_edge(START, "nodes_generator")
        gb.add_conditional_edges(
            "nodes_generator",
            NodesGenerationAgent.should_regenerate,  # returns True/False
            {True: "nodes_generator", False: END},
        )
        return gb.compile(checkpointer=checkpointer, name="Nodes Generation Agent")


if __name__ == "__main__":
    graph = NodesGenerationAgent.get_graph()
    g = graph.get_graph().draw_ascii()
    print(g)
