import pytest
from src.agent.DiscussionSummaryPerTopic import PerTopicDiscussionSummaryGenerationAgent as DSPT
from src.schema.agent_schema import AgentInternalState

@pytest.mark.asyncio
async def test_dspt_generator_and_policy(monkeypatch, inp, summary, topics, dspt):
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
        generated_summary=summary,
        interview_topics=topics,
    )

    # Stub inner per-topic graph to always return each DiscussionTopic from dspt (by index)
    class DSStub:
        def __init__(self, payloads):
            self.payloads = payloads
            self.i = 0
        async def ainvoke(self, _):
            p = self.payloads[self.i]
            self.i += 1
            return {"final_response": p}

    monkeypatch.setattr(DSPT, "_per_topic_graph", DSStub(dspt.discussion_topics))

    state = await DSPT.discussion_summary_per_topic_generator(state)
    assert state.discussion_summary_per_topic is not None
    got_names = {x.topic for x in state.discussion_summary_per_topic.discussion_topics}
    exp_names = {t.topic for t in topics.interview_topics}
    assert got_names == exp_names

    # should_regenerate should be False since names match
    regen = await DSPT.should_regenerate(state)
    assert regen is False
