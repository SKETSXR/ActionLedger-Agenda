from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState
from string import Template

from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import CollectiveInterviewTopicSchema
from ..schema.input_schema import SkillTreeSchema
from ..prompt.topic_generation_agent_prompt import TOPIC_GENERATION_AGENT_PROMPT
from ..model_handling import llm_tg
from src.mongo_tools import get_mongo_tools
from ..logging_tools import get_tool_logger, log_tool_activity, log_retry_iteration


# Global retry counter used only for logging iteration counts.
count = 1


# ---------------- Logging ----------------
AGENT_NAME = "topic_generation_agent"
LOG_DIR = "logs"
LOGGER = get_tool_logger(AGENT_NAME, log_dir=LOG_DIR, backup_count=365)


# ---------------- Inner ReAct state for Mongo loop ----------------
class _MongoAgentState(MessagesState):
    """State container for the inner Mongo-enabled ReAct loop."""
    final_response: CollectiveInterviewTopicSchema


class TopicGenerationAgent:
    """
    Generates interview topics from a prepared summary by running a small
    ReAct-style tool-using loop (for Mongo tools) and then coercing the final
    assistant content into a typed schema.
    """

    # LLMs & tools (All required class attributes are created once)
    llm_tg = llm_tg
    MONGO_TOOLS = get_mongo_tools(llm=llm_tg)
    _AGENT_MODEL = llm_tg.bind_tools(MONGO_TOOLS)
    _STRUCTURED_MODEL = llm_tg.with_structured_output(
        CollectiveInterviewTopicSchema, method="function_calling"
    )

    # ---------- Inner graph nodes ----------
    @staticmethod
    def _agent_node(state: _MongoAgentState):
        """
        Invoke the tool-enabled model. LangGraph handles actual tool execution
        via the ToolNode edge when tool calls are present.
        """
        log_tool_activity(
            messages=state["messages"],
            ai_msg=None,
            agent_name=AGENT_NAME,
            logger=LOGGER,
            header="Topic Generation Tool Activity",
            pretty_json=True,
        )
        ai = TopicGenerationAgent._AGENT_MODEL.invoke(state["messages"])
        return {"messages": [ai]}

    @staticmethod
    def _respond_node(state: _MongoAgentState):
        """
        Take the last non-tool-call AI message content, and coerce it into the
        expected structured schema using the structured-output model.
        """
        msgs = state["messages"]

        # Prefer the most recent assistant message without tool calls;
        # fall back to the most recent assistant message; finally to the last msg.
        ai_content = None
        for m in reversed(msgs):
            if getattr(m, "type", None) in ("ai", "assistant"):
                if not getattr(m, "tool_calls", None):
                    ai_content = m.content
                    break

        if ai_content is None:
            for m in reversed(msgs):
                if getattr(m, "type", None) in ("ai", "assistant"):
                    ai_content = m.content
                    break

        if ai_content is None:
            ai_content = msgs[-1].content

        final_obj = TopicGenerationAgent._STRUCTURED_MODEL.invoke(
            [HumanMessage(content=ai_content)]
        )
        return {"final_response": final_obj}

    @staticmethod
    def _should_continue(state: _MongoAgentState):
        """
        Router: if the last assistant message planned tool calls, continue to tools.
        Otherwise, proceed to respond (coerce to schema).
        """
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            log_tool_activity(
                messages=state["messages"],
                ai_msg=last,
                agent_name=AGENT_NAME,
                logger=LOGGER,
                header="Topic Generation Tool Activity",
                pretty_json=True,
            )
            return "continue"
        return "respond"

    # ---------- Compile inner ReAct graph once ----------
    _workflow = StateGraph(_MongoAgentState)
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
    _mongo_graph = _workflow.compile()

    # ---------------- Graph node: topic generation ----------------
    @staticmethod
    async def topic_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Build the system prompt from the generated summary and accumulated
        feedback, run the inner Mongo ReAct loop, and store the typed topics
        on the agent state.
        """
        if not state.generated_summary:
            raise ValueError("Summary cannot be null.")

        interview_topics_feedback = (
            state.interview_topics_feedback.feedback
            if state.interview_topics_feedback is not None
            else ""
        )
        state.interview_topics_feedbacks += "\n" + interview_topics_feedback + "\n"

        class AtTemplate(Template):
            # Take @placeholders in the prompt template as input
            delimiter = "@"

        tpl = AtTemplate(TOPIC_GENERATION_AGENT_PROMPT)
        content = tpl.substitute(
            generated_summary=state.generated_summary.model_dump_json(),
            interview_topics_feedbacks=state.interview_topics_feedbacks,
            thread_id=state.id,
        )

        messages = [
            SystemMessage(content=content),
            HumanMessage(content="Based on the instructions, please start the process."),
        ]

        result = await TopicGenerationAgent._mongo_graph.ainvoke({"messages": messages})
        state.interview_topics = result["final_response"]
        return state

    # ---------------- Graph router: should regenerate? ----------------
    @staticmethod
    async def should_regenerate(state: AgentInternalState) -> bool:
        global count

        def _canon(s: str) -> str:
            return (s or "").strip().lower()

        # depth-agnostic leaf gatherers (recursive) + skills with a must-only priority view
        def _all_leaves(root: SkillTreeSchema) -> list[SkillTreeSchema]:
            leaves: list[SkillTreeSchema] = []
            def dfs(n: SkillTreeSchema):
                if not getattr(n, "children", None):
                    leaves.append(n)
                    return
                for c in (n.children or []):
                    dfs(c)
            dfs(root)
            return leaves

        def _must_leaves(root: SkillTreeSchema) -> list[SkillTreeSchema]:
            return [l for l in _all_leaves(root) if getattr(l, "priority", None) == "must"]

        all_skill_leaves = [_canon(leaf.name) for leaf in _all_leaves(state.skill_tree)]
        skills_priority_must = [_canon(leaf.name) for leaf in _must_leaves(state.skill_tree)]

        # Gather all unique focus-area skill names from the generated topics.
        focus_area_list: list[str] = []
        for t in state.interview_topics.interview_topics:
            for i in t.focus_area:
                x = i.model_dump()
                v = _canon(x.get("skill", ""))
                if v and v not in focus_area_list:
                    focus_area_list.append(v)

        # 1) total question count must match summary total question count
        total_questions_sum = sum(t.total_questions for t in state.interview_topics.interview_topics)
        if total_questions_sum != state.generated_summary.total_questions:
            log_retry_iteration(
                agent_name=AGENT_NAME, iteration=count, reason="Total questions mismatch",
                logger=LOGGER, pretty_json=True,
                extra={"got_total": total_questions_sum, "target_total": state.generated_summary.total_questions},
            )
            count += 1
            return False  # retry

        # 2) every focus skill must exist in leaves
        leaf_set = set(all_skill_leaves)
        for s in focus_area_list:
            if s not in leaf_set:
                log_retry_iteration(
                    agent_name=AGENT_NAME, iteration=count, reason="Invalid focus skill",
                    logger=LOGGER, pretty_json=True, extra={"skill": s},
                )
                count += 1
                return False  # retry

        # 3) every MUST leaf must appear at least once
        missing = sorted(set(skills_priority_must) - set(focus_area_list))
        if missing:
            fb = state.interview_topics_feedback.feedback if state.interview_topics_feedback else ""
            fb += (
                "Please keep the topic set as is irrespective of below instructions:\n"
                f"```\n{state.interview_topics.model_dump()}\n```\n"
                "But add the list of missing `must` priority skills below to the focus areas of the last topic "
                "(General Skill Assessment):\n"
                + ", ".join(missing)
            )
            state.interview_topics_feedback = {"satisfied": False, "feedback": fb}

            log_retry_iteration(
                agent_name=AGENT_NAME, iteration=count, reason="Missing MUST skills",
                logger=LOGGER, pretty_json=True, extra={"missing_must": missing},
            )
            count += 1
            return False  # retry

        return True  # satisfied

    # ---------------- Outer main topic generation graph ----------------
    @staticmethod
    def get_graph(checkpointer=None):
        """
        Build the Topic Generation graph:
        START -> topic_generator -> (should_regenerate ? END : topic_generator)
        """
        graph_builder = StateGraph(state_schema=AgentInternalState)
        graph_builder.add_node("topic_generator", TopicGenerationAgent.topic_generator)
        graph_builder.add_edge(START, "topic_generator")
        graph_builder.add_conditional_edges(
            "topic_generator",
            TopicGenerationAgent.should_regenerate,
            {True: END, False: "topic_generator"},
        )
        return graph_builder.compile(
            checkpointer=checkpointer, name="Topic Generation Agent"
        )


if __name__ == "__main__":
    graph = TopicGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
