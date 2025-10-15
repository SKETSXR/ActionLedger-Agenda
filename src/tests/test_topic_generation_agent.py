import pytest

# Import the class and the AsyncGraphStub fixture
from src.agent.TopicGenerationAgent import TopicGenerationAgent
from src.schema.agent_schema import AgentInternalState
from src.tests.conftest import AsyncGraphStub


@pytest.mark.asyncio
async def test_topic_generator_and_should_regenerate(monkeypatch, inp, summary, topics):
    # Build state skeleton (skip Agenda input_formatter for brevity)
    state = AgentInternalState(
        mongo_client="mongodb://localhost:27017",
        mongo_db="agenda_db",
        mongo_jd_collection="jd",
        mongo_cv_collection="cv",
        mongo_skill_tree_collection="skill_tree",
        mongo_summary_collection="summary",
        mongo_question_guidelines_collection="question_guidelines",
        id="thread-test-1",
        job_description=inp.job_description,
        skill_tree=inp.skill_tree,
        candidate_profile=inp.candidate_profile,
        question_guidelines=inp.question_guidelines,
        generated_summary=summary,
    )

    # Stub the compiled inner graph (the agent calls _get_inner_graph().ainvoke({...}))
    stub = AsyncGraphStub({"final_response": topics})
    monkeypatch.setattr(
        TopicGenerationAgent, "_get_inner_graph", lambda: stub, raising=True
    )

    # Run the node
    state = await TopicGenerationAgent.topic_generator(state)
    assert state.interview_topics is not None
    assert len(state.interview_topics.interview_topics) == 2

    # should_regenerate checks:
    # 1) totals must match summary.total_questions (we set 4 + 2 = 6) → OK
    # 2) focus_area skills must be leaves → OK due to fixtures
    # 3) all MUST leaves must appear at least once → fixtures cover that
    ok = await TopicGenerationAgent.should_regenerate(state)
    assert ok is True  # satisfied → graph will END
