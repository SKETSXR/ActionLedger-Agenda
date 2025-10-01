from typing import Annotated, Optional, List, Union, Literal
from pydantic import BaseModel, Field


# ============================== SUMMARY (JD + Candidate) ============================== #
class JobRequirementsSummarySchema(BaseModel):
    """Company expectations and basic requirements extracted from the JD."""

    company_expectations_tech: Annotated[
        str,
        Field(
            ...,
            description="Technical expectations of the company",
            examples=[
                "Proficiency in programming languages: C, C++, Visual C++, MFC, C#, and WPF. "
                "Strong debugging and analytical skills. Knowledge of Component Object Model (COM/DCOM)."
            ],
        ),
    ]
    about_company_or_product: Annotated[
        str,
        Field(
            ...,
            description="About the Company / Product",
            examples=["The company background is not provided."],
        ),
    ]
    fundamental_knowledge: Annotated[
        str,
        Field(
            ...,
            description="Education qualification requirements if any",
            examples=["B.Tech or M.Tech in any field."],
        ),
    ]


class ProjectReasoningSummarySchema(BaseModel):
    """Short reasoning summary per project: what, how, stack, and a brief walkthrough."""
    what_done: Annotated[str, Field(..., description="What was built/achieved")]
    how_done: Annotated[str, Field(..., description="How it was implemented (approach/architecture)")]
    tech_stack: Annotated[str, Field(..., min_items=1, description="Technologies used")]
    walkthrough: Annotated[str, Field(..., description="Brief step-by-step of how each particular tech stack was used")]


class CandidateProjectSummarySchema(BaseModel):
    """Aggregate of project-wise reasoning summaries."""
    projectwise_summary: List[ProjectReasoningSummarySchema] = Field(
        ..., description="List of project wise summary"
    )


# ============================== ANNOTATED SKILL TREE (EVIDENCE) ============================== #
class SkillLeaf(BaseModel):
    """Leaf skill with normalized weight, priority, and a short evidence comment."""
    name: str = Field(..., description="Leaf skill name (verbatim)")
    weight: float = Field(..., description="Normalized in [0,1]")
    priority: Literal["must", "high", "low"] = Field(..., description="Priority")
    comment: str = Field(..., description="Short evidence/comment for this skill")


class DomainNode(BaseModel):
    """Domain node containing leaf skills and optional comment."""
    name: str = Field(..., description="Domain name")
    weight: float = Field(..., description="Normalized in [0,1]")
    priority: Literal["must", "high", "low"]
    comment: Optional[str] = None
    children: List[SkillLeaf] = Field(..., description="Non-empty list of leaf skills")


class AnnotatedSkillTreeSummarySchema(BaseModel):
    """Root of the annotated skill tree summarization with domains and leaves."""
    name: str = Field(..., description="Root label, e.g., 'L3 (Software Engineer 2)'")
    weight: float = Field(..., description="Usually 1.0")
    priority: Literal["must", "high", "low"]
    comment: Optional[str] = None
    children: List[DomainNode] = Field(..., description="Non-empty list of domains")


class DomainsToAssessListSchema(BaseModel):
    """Flat list of domains to assess (derived view)."""

    class Domain(BaseModel):
        name: str
        weight: float
        priority: Literal["must", "high", "low"]

    domains: List[Domain]


class TotalQuestionsSchema(BaseModel):
    """Total planned question count for the entire interview."""
    total_questions: Annotated[
        int,
        Field(
            ...,
            description="Total planned questions in the entire interview",
            examples=[18],
        ),
    ]


class GeneratedSummarySchema(TotalQuestionsSchema):
    """
    Consolidated summary produced by the SummaryGenerationAgent:
    - JD requirements
    - Candidate project reasoning
    - Annotated skill tree
    - Domains to assess
    - Total question count (inherited)
    """
    job_requirements: JobRequirementsSummarySchema
    candidate_project_summary: CandidateProjectSummarySchema
    annotated_skill_tree_T: AnnotatedSkillTreeSummarySchema
    domains_assess_D: DomainsToAssessListSchema


# ============================== TOPICS & FEEDBACK ============================== #
class FocusAreaSchema(BaseModel):
    """Focus skill (verbatim leaf) and a short guideline for probing it."""
    skill: str = Field(..., description="Verbatim leaf skill from annotated skill tree")
    guideline: str = Field(..., description="Brief guideline on what to focus on for this skill")


class TopicSchema(BaseModel):
    """One discussion topic with rationale, focus areas, references, and planned count."""
    topic: Annotated[str, Field(..., description="Short name of the discussion topic")]
    why_this_topic: Annotated[str, Field(..., description="A short reason for why this discussion topic has been chosen")]
    focus_area: List[FocusAreaSchema] = Field(
        ...,
        description="List of focus skills with guidelines (each item is {skill, guideline})",
        examples=[
            [
                {
                    "skill": "System design (throughput, latency, fault tolerance)",
                    "guideline": "Probe trade-offs in throughput/latency/fault tolerance for the candidate's context.",
                },
                {
                    "skill": "Node.js (advanced services, concurrency, resilience)",
                    "guideline": "Discuss patterns for concurrency, resilience, retries, and graceful degradation.",
                },
                {
                    "skill": "GraphQL & REST (scaling & governance)",
                    "guideline": "Check versioning, schema governance, backward compatibility, and rate limiting.",
                },
                {
                    "skill": "Postgres: indexing, query tuning, partitioning",
                    "guideline": "Evaluate indexing choices, query plans, and partitioning strategies.",
                },
            ]
        ],
    )
    necessary_reference_material: Annotated[str, Field(..., description="Reference material for this topic")]
    total_questions: Annotated[int, Field(..., description="Planned question count")]


class CollectiveInterviewTopicSchema(BaseModel):
    """Collection of interview topics."""
    interview_topics: List[TopicSchema] = Field(..., description="List of interview topics")


class CollectiveInterviewTopicFeedbackSchema(BaseModel):
    """Feedback payload for iteratively refining the topics."""
    satisfied: bool
    feedback: str


# ============================== PER-TOPIC DISCUSSION SUMMARY ============================== #
class DiscussionSummaryPerTopicSchema(BaseModel):
    """
    Per-topic discussion plan broken into Opening, Direct, and DeepDive segments.
    Each segment lists description/guidelines/focus-areas/references.
    """

    class Opening(BaseModel):
        type: str
        description: str
        guidelines: str
        focus_areas: List[str]
        reference_sources: List[str]

    class DirectQuestion(BaseModel):
        type: str
        description: str
        guidelines: str
        focus_areas: List[str]
        reference_sources: List[str]

    class DeepDive(BaseModel):
        type: str
        description: str
        guidelines: str
        focus_areas: List[str]
        reference_sources: List[str]

    class DiscussionTopic(BaseModel):
        topic: str
        sequence: List[
            Union[
                "DiscussionSummaryPerTopicSchema.Opening",
                "DiscussionSummaryPerTopicSchema.DirectQuestion",
                "DiscussionSummaryPerTopicSchema.DeepDive",
            ]
        ]
        guidelines: str
        focus_areas_covered: List[str]
        reference_material: List[str]

    discussion_topics: List[DiscussionTopic]


# ============================== NODES (TOPIC FLOW GRAPH) ============================== #
QuestionType = Literal["Direct", "Deep Dive"]


class NodeSchema(BaseModel):
    """
    One node/question in the topic flow:
    - Direct nodes carry the actual question text.
    - Deep Dive nodes reference context, skills, and optional thresholds.
    """
    id: int = Field(..., ge=1)
    question_type: QuestionType
    question: Optional[str] = None  # Required if question_type is "Direct"
    graded: bool
    next_node: Optional[int] = Field(None, ge=1)
    context: str = Field(..., min_length=1)
    skills: List[str] = Field(..., min_items=1)
    total_question_threshold: Optional[int] = Field(None, ge=2)
    question_guidelines: Optional[str] = None


class TopicWithNodesSchema(BaseModel):
    """A topic with its ordered list of nodes."""
    topic: str
    nodes: List[NodeSchema] = Field(..., min_items=1)


class NodesSchema(BaseModel):
    """Container for topic-node flows across all topics."""
    topics_with_nodes: List[TopicWithNodesSchema] = Field(..., min_items=1)


# ============================== QA BLOCKS (QUESTION SETS) ============================== #
QType = Literal["New Question", "Counter Question"]
QDiff = Literal["Easy", "Medium", "Hard"]
QCountType = Literal["Twist", "Interrogatory"]


class QAItem(BaseModel):
    """One question item within a block, with difficulty and example questions."""
    qa_id: str = Field(..., description="QA identifier like 'QA1'")
    q_type: QType
    q_difficulty: QDiff
    counter_type: Optional[QCountType] = None
    example_questions: List[str] = Field(..., min_items=5, max_items=5)


class QABlock(BaseModel):
    """A block groups related QA items under a single guideline."""
    block_id: str = Field(..., description="Block identifier like 'B1'")
    guideline: str = Field(..., min_length=1)
    qa_items: List[QAItem] = Field(..., min_items=7, max_items=7)


class QASet(BaseModel):
    """All blocks generated for a single topic."""
    topic: str = Field(..., min_length=1)
    qa_blocks: List[QABlock] = Field(..., min_items=1)


class QASetsSchema(BaseModel):
    """Top-level collection of QA sets across topics."""
    qa_sets: List[QASet] = Field(..., min_items=1)


# ============================== OUTPUT ROOT ============================== #
class OutputSchema(BaseModel):
    """Final pipeline output bundle."""
    summary: GeneratedSummarySchema
    interview_topics: CollectiveInterviewTopicSchema
    discussion_summary_per_topic: DiscussionSummaryPerTopicSchema
    nodes: NodesSchema
    qa_blocks: QASetsSchema
