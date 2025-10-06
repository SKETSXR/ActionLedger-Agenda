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

    # Stub inner graph to return each TopicWithNodesSchema sequentially
    class NodesStub:
        def __init__(self, payloads):
            self.payloads = payloads
            self.i = 0
        async def ainvoke(self, _):
            p = self.payloads[self.i]
            self.i += 1
            return {"final_response": p}

    monkeypatch.setattr(NA, "_nodes_graph", NodesStub(nodes.topics_with_nodes))

    # ensure per-topic minimum (Opening + Direct + one Deep Dive with threshold >=2)
    state.interview_topics.interview_topics[1].total_questions = 4

    state = await NA.nodes_generator(state)
    assert state.nodes is not None
    assert len(state.nodes.topics_with_nodes) == 2

    regen = await NA.should_regenerate(state)
    assert regen is False
