import pytest
from src.agent.QABlocksAgent import QABlockGenerationAgent as QA
from src.schema.agent_schema import AgentInternalState
from src.schema.output_schema import QASetsSchema, QASet, QABlock, QAItem


@pytest.mark.asyncio
async def test_qablocks_generator_and_policy(monkeypatch, inp, summary, topics, dspt, nodes):
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
        nodes=nodes,
    )

    # For topic1: 2 deep dives -> need 2 QA blocks
    # For topic2: 1 deep dive -> need 1 QA block
    def _mk_block(i: int) -> QABlock:
        # exactly 7 items, no Easy counters
        items = [
            QAItem(
                qa_id=f"QA{i}-{k}",
                q_type="New Question",
                q_difficulty="Medium",
                counter_type=None,
                example_questions=["a", "b", "c", "d", "e"],
            )
            for k in range(1, 6)
        ] + [
            QAItem(
                qa_id=f"QA{i}-6",
                q_type="Counter Question",
                q_difficulty="Medium",
                counter_type="Twist",
                example_questions=["a", "b", "c", "d", "e"],
            ),
            QAItem(
                qa_id=f"QA{i}-7",
                q_type="Counter Question",
                q_difficulty="Hard",
                counter_type="Interrogatory",
                example_questions=["a", "b", "c", "d", "e"],
            ),
        ]
        return QABlock(block_id=f"B{i}", guideline="follow guides", qa_items=items)

    topic1_set = QASet(
        topic=topics.interview_topics[0].topic,
        qa_blocks=[_mk_block(1), _mk_block(2)],
    )
    topic2_set = QASet(
        topic=topics.interview_topics[1].topic,
        qa_blocks=[_mk_block(3)],
    )

    # Stub the per-topic generator to return the single QA set (dict) + no error
    payloads = [topic1_set.model_dump(), topic2_set.model_dump()]
    idx = {"i": 0}

    async def fake_gen_for_topic(topic_name, discussion_summary_json, deep_dive_nodes_json, thread_id, qa_error=""):
        i = idx["i"]
        idx["i"] = i + 1
        # The real method returns (one_set_dict, error_string)
        return payloads[i], ""

    monkeypatch.setattr(QA, "_gen_for_topic", fake_gen_for_topic)

    # Run generator
    state = await QA.qablock_generator(state)
    assert state.qa_blocks is not None
    assert isinstance(state.qa_blocks, QASetsSchema)
    assert len(state.qa_blocks.qa_sets) == 2
    assert state.qa_error == "" or isinstance(state.qa_error, str)

    # Policy should accept (no retry)
    regen = await QA.should_regenerate(state)
    assert regen is False
