import pytest

# Import the module itself so we can patch its module-level _llm_client
import src.agent.SummaryGenerationAgent as SGMOD
from src.agent.SummaryGenerationAgent import SummaryGenerationAgent as SGA
from src.schema.agent_schema import AgentInternalState


# ---- minimal fake LLM that mimics the structured-output chain ----
class _FakeLLM:
    """
    Emulates:
      _llm_client.with_structured_output(GeneratedSummarySchema).ainvoke(messages)
    """
    def __init__(self, return_value):
        self._return_value = return_value

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, _messages):
        return self._return_value


@pytest.mark.asyncio
async def test_summary_generator_populates_state(monkeypatch, inp, summary):
    """
    Unit-test the node function directly by stubbing the module-level _llm_client.
    """
    # Build a minimal AgentInternalState (mirror what input_formatter would produce)
    state = AgentInternalState(
        mongo_client="mongodb://localhost:27017", mongo_db="agenda_db",
        mongo_jd_collection="jd", mongo_cv_collection="cv",
        mongo_skill_tree_collection="skill_tree", mongo_summary_collection="summary",
        mongo_question_guidelines_collection="question_guidelines",
        id="thread-test-1",
        job_description=inp.job_description,
        skill_tree=inp.skill_tree,
        candidate_profile=inp.candidate_profile,
        question_guidelines=inp.question_guidelines,
    )

    # Patch the module-level client the agent uses internally
    monkeypatch.setattr(SGMOD, "_llm_client", _FakeLLM(summary), raising=True)

    # Run the node
    state = await SGA.summary_generator(state)

    assert state.generated_summary is not None
    assert state.generated_summary.total_questions == summary.total_questions
    assert state.generated_summary.job_requirements.about_company_or_product == inp.job_description.company_background
    assert (
        "Node" in state.generated_summary.job_requirements.company_expectations_tech
        or "Postgres" in state.generated_summary.job_requirements.company_expectations_tech
    )


@pytest.mark.asyncio
async def test_summary_graph_integration(monkeypatch, inp, summary):
    """
    Integration-lite: run the compiled graph (single node) with the fake LLM.
    """
    # Prepare state
    state = AgentInternalState(
        mongo_client="mongodb://localhost:27017", mongo_db="agenda_db",
        mongo_jd_collection="jd", mongo_cv_collection="cv",
        mongo_skill_tree_collection="skill_tree", mongo_summary_collection="summary",
        mongo_question_guidelines_collection="question_guidelines",
        id="thread-test-1",
        job_description=inp.job_description,
        skill_tree=inp.skill_tree,
        candidate_profile=inp.candidate_profile,
        question_guidelines=inp.question_guidelines,
    )

    # Stub the module-level client used by invoke_llm_with_retry
    monkeypatch.setattr(SGMOD, "_llm_client", _FakeLLM(summary), raising=True)

    graph = SGA.get_graph()
    out_state = await graph.ainvoke(state)

    # handle both dict and Pydantic state
    if isinstance(out_state, dict):
        assert out_state.get("generated_summary") is not None
        assert out_state["generated_summary"].total_questions == summary.total_questions
    else:
        assert out_state.generated_summary is not None
        assert out_state.generated_summary.total_questions == summary.total_questions
