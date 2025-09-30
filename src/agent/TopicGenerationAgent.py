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
        """
        Decide whether to regenerate topics based on:
        1) Total question count matching the generated summary's target.
        2) All focus-area skills belonging to the leaf set of the skill tree.
        3) Ensuring all 'must' priority leaf skills are included at least once.
        When missing MUST skills, feedback is injected to add them to the
        'General Skill Assessment' topic's focus areas.
        """
        global count

        def level3_leavesp(root: SkillTreeSchema) -> list[SkillTreeSchema]:
            """Collect true leaves with priority == 'must' from the 3rd level."""
            if not root.children:
                return []
            skills_priority_must: list[SkillTreeSchema] = []
            for domain in root.children:
                for leaf in (domain.children or []):
                    if not leaf.children and leaf.priority == "must":
                        skills_priority_must.append(leaf)
            return skills_priority_must

        def level3_leaves(root: SkillTreeSchema) -> list[SkillTreeSchema]:
            """Collect all true leaves from the 3rd level."""
            if not root.children:
                return []
            leaves: list[SkillTreeSchema] = []
            for domain in root.children:
                for leaf in (domain.children or []):
                    if not leaf.children:
                        leaves.append(leaf)
            return leaves

        all_skill_leaves = [leaf.name for leaf in level3_leaves(state.skill_tree)]
        skills_priority_must = [leaf.name for leaf in level3_leavesp(state.skill_tree)]

        # Gather all unique focus-area skill names from the generated topics.
        focus_area_list: list[str] = []
        for t in state.interview_topics.interview_topics:
            for i in t.focus_area:
                for key in i.model_dump():
                    if key == "skill":
                        x = i.model_dump()
                        if x["skill"] not in focus_area_list:
                            focus_area_list.append(x["skill"])

        # Validate total questions count.
        total_questions_sum = sum(
            t.total_questions for t in state.interview_topics.interview_topics
        )
        if total_questions_sum != state.generated_summary.total_questions:
            log_retry_iteration(
                agent_name=AGENT_NAME,
                iteration=count,
                reason="Total questions mismatch",
                logger=LOGGER,
                pretty_json=True,
                extra={
                    "got_total": total_questions_sum,
                    "target_total": state.generated_summary.total_questions,
                },
            )
            count += 1
            return False

        # Validate that every focus-area skill exists in the leaf set.
        for s in focus_area_list:
            if s not in all_skill_leaves:
                log_retry_iteration(
                    agent_name=AGENT_NAME,
                    iteration=count,
                    reason="Invalid focus skill",
                    logger=LOGGER,
                    pretty_json=True,
                    extra={"skill": s},
                )
                count += 1
                return False

        # Ensure all MUST-priority leaves are represented at least once.
        missing_must = ""
        for skill in set(skills_priority_must):
            if skill not in set(focus_area_list):
                missing_must += ", " + skill

        if missing_must != "":
            feedback = ""
            if state.interview_topics_feedback is not None:
                feedback = state.interview_topics_feedback.feedback
            feedback += (
                "Please keep the topic set as it is irresepective of below instructions: "
                f"```\n{state.interview_topics.model_dump()}```\n "
                "But add the list of missing `must` priority skills in this as per the \n"
                f"{missing_must}\n "
                "to the focus areas of the last topic which being General Skill Assessment"
            )
            state.interview_topics_feedback = {"satisfied": False, "feedback": feedback}

            log_retry_iteration(
                agent_name=AGENT_NAME,
                iteration=count,
                reason="Missing MUST skills",
                logger=LOGGER,
                pretty_json=True,
                extra={"missing_must": missing_must},
            )
            count += 1
            return False

        return True

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
