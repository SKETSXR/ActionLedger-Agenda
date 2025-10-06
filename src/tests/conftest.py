# --- ensure project root is on sys.path ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root (two levels up from src/tests)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import mongomock
from typing import Any, Dict
from types import SimpleNamespace

from src.schema.input_schema import (
    SkillTreeSchema, JobDescriptionSchema, CandidateProfileSchema,
    ProjectSchema, ExperienceSchema, QuestionGuidelinesSchema, QuestionGuidelinesCompleteSchema, InputSchema
)
from src.schema.output_schema import (
    GeneratedSummarySchema, JobRequirementsSummarySchema, CandidateProjectSummarySchema,
    ProjectReasoningSummarySchema, AnnotatedSkillTreeSummarySchema, DomainsToAssessListSchema,
    CollectiveInterviewTopicSchema, TopicSchema, FocusAreaSchema,
    DiscussionSummaryPerTopicSchema, NodesSchema, TopicWithNodesSchema, NodeSchema,
    QASetsSchema, QASet, QABlock, QAItem
)

# ---------- tiny async stub with ainvoke ----------
class AsyncGraphStub:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
    async def ainvoke(self, _):
        return self.payload

@pytest.fixture
def skill_tree_3level():
    # root -> 2 domains -> leaves (with "must" priorities)
    return SkillTreeSchema(
        name="Root", weight=1.0, priority="must",
        children=[
            SkillTreeSchema(
                name="Backend", weight=0.5, priority="high",
                children=[
                    SkillTreeSchema(name="Node.js (advanced services, concurrency, resilience)", weight=0.25, priority="must"),
                    SkillTreeSchema(name="GraphQL & REST (scaling & governance)", weight=0.25, priority="must"),
                ],
            ),
            SkillTreeSchema(
                name="Databases", weight=0.5, priority="high",
                children=[
                    SkillTreeSchema(name="Postgres: indexing, query tuning, partitioning", weight=0.25, priority="must"),
                    SkillTreeSchema(name="System design (throughput, latency, fault tolerance)", weight=0.25, priority="high"),
                ],
            ),
        ],
    )

@pytest.fixture
def jd():
    return JobDescriptionSchema(
        job_role="Backend Developer",
        company_background="Serious widgets",
        fundamental_knowledge="B.Tech preferred"
    )

@pytest.fixture
def candidate():
    return CandidateProfileSchema(
        skills=["Node.js","Postgres","GraphQL","System Design"],
        projects=[ProjectSchema(id="P1", title="Orders", description="Order mgmt")],
        experience=[ExperienceSchema(id="E1", company="Acme", title="SWE", description="APIs")]
    )

@pytest.fixture
def qg():
    return QuestionGuidelinesCompleteSchema(
        question_guidelines=[
            QuestionGuidelinesSchema(question_type_name="Direct", question_guidelines="Be concise"),
            QuestionGuidelinesSchema(question_type_name="Deep Dive", question_guidelines="Probe tradeoffs"),
        ]
    )

@pytest.fixture
def inp(skill_tree_3level, jd, candidate, qg):
    return InputSchema(
        job_description=jd,
        skill_tree=skill_tree_3level,
        candidate_profile=candidate,
        question_guidelines=qg
    )

@pytest.fixture
def summary(skill_tree_3level, jd):
    return GeneratedSummarySchema(
        total_questions=6,
        job_requirements=JobRequirementsSummarySchema(
            company_expectations_tech="Node.js, GraphQL, Postgres",
            about_company_or_product=jd.company_background,
            fundamental_knowledge=jd.fundamental_knowledge
        ),
        candidate_project_summary=CandidateProjectSummarySchema(
            projectwise_summary=[ProjectReasoningSummarySchema(
                what_done="Built orders", how_done="FastAPI+PG", tech_stack="FastAPI, Postgres",
                walkthrough="CRUD -> queues -> retries"
            )]
        ),
        annotated_skill_tree_T=AnnotatedSkillTreeSummarySchema(
            name="Root", weight=1.0, priority="must", children=[]
        ),
        domains_assess_D=DomainsToAssessListSchema(
            domains=[DomainsToAssessListSchema.Domain(name="Backend", weight=0.5, priority="high")]
        )
    )

@pytest.fixture
def topics(skill_tree_3level):
    # 2 topics, each total_questions >= 4 (meets NodesGen guard)
    return CollectiveInterviewTopicSchema(
        interview_topics=[
            TopicSchema(
                topic="E-commerce Platform Scalability",
                why_this_topic="Covers peak traffic + DB",
                focus_area=[
                    FocusAreaSchema(skill="Postgres: indexing, query tuning, partitioning", guideline="indexing choices"),
                    FocusAreaSchema(skill="Node.js (advanced services, concurrency, resilience)", guideline="concurrency/resilience"),
                ],
                necessary_reference_material="RFCs",
                total_questions=4,
            ),
            TopicSchema(
                topic="Designing a Scalable Content Platform",
                why_this_topic="Covers API governance + system design",
                focus_area=[
                    FocusAreaSchema(skill="GraphQL & REST (scaling & governance)", guideline="versioning, rate limiting"),
                    FocusAreaSchema(skill="System design (throughput, latency, fault tolerance)", guideline="tradeoffs"),
                ],
                necessary_reference_material="Docs",
                total_questions=2,  # will cause mismatch unless summary.total == 6 (4 + 2)
            ),
        ]
    )

@pytest.fixture
def dspt(topics):
    # minimal aligned discussion summaries (topic names must match input)
    return DiscussionSummaryPerTopicSchema(
        discussion_topics=[
            DiscussionSummaryPerTopicSchema.DiscussionTopic(
                topic=t.topic,
                sequence=[
                    DiscussionSummaryPerTopicSchema.Opening(
                        type="Opening", description="Background", guidelines="short", focus_areas=[], reference_sources=[]
                    ),
                    DiscussionSummaryPerTopicSchema.DirectQuestion(
                        type="Direct", description="Direct Q", guidelines="direct", focus_areas=[], reference_sources=[]
                    ),
                    DiscussionSummaryPerTopicSchema.DeepDive(
                        type="Deep Dive", description="Deep probe", guidelines="deep", focus_areas=[], reference_sources=[]
                    ),
                ],
                guidelines="global",
                focus_areas_covered=[fa.skill for fa in t.focus_area],
                reference_material=["RFCs"]
            )
            for t in topics.interview_topics
        ]
    )

@pytest.fixture
def nodes(topics):
    # each topic: 1 Direct + N Deep Dive (here choose 2 deep dives for topic1, 1 for topic2)
    return NodesSchema(
        topics_with_nodes=[
            TopicWithNodesSchema(
                topic=topics.interview_topics[0].topic,
                nodes=[
                    NodeSchema(id=1, question_type="Direct", question="What is...", graded=True, next_node=2,
                               context="ctx", skills=["Postgres: indexing, query tuning, partitioning"]),
                    NodeSchema(id=2, question_type="Deep Dive", graded=True, next_node=3, context="ctx",
                               skills=["Node.js (advanced services, concurrency, resilience)"], total_question_threshold=2),
                    NodeSchema(id=3, question_type="Deep Dive", graded=True, next_node=None, context="ctx",
                               skills=["Postgres: indexing, query tuning, partitioning"], total_question_threshold=2),
                ],
            ),
            TopicWithNodesSchema(
                topic=topics.interview_topics[1].topic,
                nodes=[
                    NodeSchema(id=1, question_type="Direct", question="Tell me...", graded=True, next_node=2,
                               context="ctx", skills=["GraphQL & REST (scaling & governance)"]),
                    NodeSchema(id=2, question_type="Deep Dive", graded=True, next_node=None, context="ctx",
                               skills=["System design (throughput, latency, fault tolerance)"], total_question_threshold=2),
                ],
            ),
        ]
    )

# simple mongomock install for any accidental imports
@pytest.fixture(autouse=True)
def _patch_pymongo(monkeypatch):
    import pymongo
    monkeypatch.setattr(pymongo, "MongoClient", mongomock.MongoClient)
    yield
