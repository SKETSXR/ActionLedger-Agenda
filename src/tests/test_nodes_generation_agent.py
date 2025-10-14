import pytest
from src.agent.NodesAgent import NodesGenerationAgent as NA
from src.schema.agent_schema import AgentInternalState


@pytest.mark.asyncio
async def test_nodes_generator_and_policy(monkeypatch, inp, summary, topics, dspt, nodes):
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
        discussion_summary_per_topic=dspt,
    )

    # Stub the INNER per-topic generator to return each TopicWithNodes in sequence
    payloads = list(nodes.topics_with_nodes)
    idx = {"i": 0}

    async def fake_gen_once(_per_topic_summary_json, _thread_id, _nodes_error):
        i = idx["i"]
        out = payloads[i]
        idx["i"] = i + 1
        return out

    monkeypatch.setattr(NA, "_gen_once", fake_gen_once)

    # (The new Nodes agent no longer enforces per-topic total_questions >= 4,
    # so we don't need to tweak the fixture anymore.)

    state = await NA.nodes_generator(state)

    assert state.nodes is not None
    assert len(state.nodes.topics_with_nodes) == 2

    # Container + per-topic schema validation should pass → no regeneration
    regen = await NA.should_regenerate(state)
    assert regen is False
