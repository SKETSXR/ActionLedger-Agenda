import pytest
from src.agent.DiscussionSummaryPerTopic import (
    PerTopicDiscussionSummaryAgent as DSPA,
    PerTopicDiscussionSummaryGenerationAgent as InnerDSPT,
)
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

    # Build a name->DiscussionTopic lookup so the stub is safe under concurrency
    by_name = {dt.topic: dt for dt in dspt.discussion_topics}

    async def fake_one_topic_call(_gs_json, topic_dict, _thread_id):
        # topic_dict comes from .model_dump() of the input topic
        tname = (
            topic_dict.get("topic")
            or topic_dict.get("name")
            or topic_dict.get("title")
            or "Unknown"
        )
        return by_name[tname]

    # Patch the inner per-topic call used by the outer generator
    monkeypatch.setattr(InnerDSPT, "_one_topic_call", fake_one_topic_call)

    # Run the OUTER generator (it fans out to inner calls concurrently)
    state = await DSPA.discussion_summary_per_topic_generator(state)

    assert state.discussion_summary_per_topic is not None
    got_names = {x.topic for x in state.discussion_summary_per_topic.discussion_topics}
    exp_names = {t.topic for t in topics.interview_topics}
    assert got_names == exp_names

    # With sets matching, router should decide not to regenerate
    regen = await DSPA.should_regenerate(state)
    assert regen is False
