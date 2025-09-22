from pydantic import BaseModel, Field
from typing import Annotated, Optional, Dict, List, Union, Literal

__all__ = [
    "GeneratedSummarySchema",
    "InterviewTopicsSchema",
    "OutputSchema"
]


class JobRequirementsSummarySchema(BaseModel):

    company_expectations_tech: Annotated[
        str,
        Field(...,
            description="Technical expectations of the company",
            examples=["Proficiency in programming languages: C, C++, Visual C++, MFC, C#, and WPF. Strong debugging and analytical skills. Knowledge of Component Object Model (COM/DCOM)."]
        )
    ]
    about_companyorproduct: Annotated[
        str,
        Field(...,
            description="About the Company / Product",
            examples=["The company background is not provided."]
        )
    ]
    fundamental_knowledge: Annotated[
        str,
        Field(...,
            description="Education qualification requirements if any",
            examples=["B.Tech or M.Tech in any field."]
        )
    ]

class ProjectReasoningSummarySchema(BaseModel):
    what_done: Annotated[str, Field(..., description="What was built/achieved")]
    how_done: Annotated[str, Field(..., description="How it was implemented (approach/architecture)")]
    tech_stack: Annotated[str, Field(..., min_items=1, description="Technologies used")]
    walkthrough: Annotated[str, Field(..., description="Brief step-by-step of how each particular tech stack was used")]

class CandidateProjectSummarySchema(BaseModel):
    projectwise_summary: List[ProjectReasoningSummarySchema] = Field(..., description="List of project wise summary")   


class AnnotatedSkillTreeSummarySchema(BaseModel):
    children: Optional[list['AnnotatedSkillTreeSummarySchema']] = Field(
        default=None,
        description="Child nodes; present for domains, empty or None for skills"
    )
    name: Annotated[
        str,
        Field(
            ...,
            description="Name of the skill or domain",
            examples=["Databases"]
        )
    ]
    weight: Annotated[
        float,
        Field(
            ...,
            description="Importance of this skill/domain in the overall evaluation. Value is normalized between 0 and 1",
            examples=[0.2]
        )
    ]
    priority: Literal["must", "high", "low"] = Field(..., description="Priority of the skill or domain", examples=["must", "high", "low"])
    
    comment: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Evidence comment required for skills (leaves), forbidden for domains",
            examples=[
                "Integrated AI-driven document processing solutions using LangChain, Python, and cloud-native architectures",
                "no such evidence"
            ]
        )
    ]


class DomainToAssessSchema(BaseModel):
    name: str = Field(..., description="Domain name, e.g., 'Machine Learning'")
    weight: float = Field(..., description="Normalized importance in [0, 1]")
    priority: Literal["must", "high", "low"] = Field(..., description="Priority of the skill or domain", examples=["must", "high", "low"])


class DomainsToAssessListSchema(BaseModel):
    domains: List[DomainToAssessSchema] = Field(
        ..., description="List of domains to be assessed"
    )

class TotalQuestionsSchema(BaseModel):
    total_questions: Annotated[
        int,
        Field(
            ...,
            description="Total planned questions in the entire interview",
            examples=[18]
        )
    ]

class GeneratedSummarySchema(TotalQuestionsSchema):
    job_requirements: JobRequirementsSummarySchema
    candidate_project_summary: CandidateProjectSummarySchema
    annotated_skill_tree_T: AnnotatedSkillTreeSummarySchema
    domains_assess_D: DomainsToAssessListSchema


class FocusAreaSchema(BaseModel):
    skill: str = Field(..., description="Verbatim leaf skill from annotated skill tree")
    guideline: str = Field(..., description="Brief guideline on what to focus on for this skill")


class TopicSchema(BaseModel):
    topic: Annotated[str, Field(..., description="Short name of the discussion topic")]
    why_this_topic: Annotated[str, Field(..., description="A short reason for why this discussion topic has been chosen")]
    # focus_area: Annotated[Dict[str, str], Field(..., description="skill -> focus on this area")]
    focus_area: List[FocusAreaSchema] = Field(
        ...,
        description="List of focus skills with guidelines (each item is {skill, guideline})",
        examples=[[
            {"skill": "System design (throughput, latency, fault tolerance)",
             "guideline": "Probe trade-offs in throughput/latency/fault tolerance for the candidate's context."},
            {"skill": "Node.js (advanced services, concurrency, resilience)",
             "guideline": "Discuss patterns for concurrency, resilience, retries, and graceful degradation."},
            {"skill": "GraphQL & REST (scaling & governance)",
             "guideline": "Check versioning, schema governance, backward compatibility, and rate limiting."},
            {"skill": "Postgres: indexing, query tuning, partitioning",
             "guideline": "Evaluate indexing choices, query plans, and partitioning strategies."}
        ]]
    )
    necessary_reference_material: Annotated[str, Field(..., description="Reference material for this topic")]
    total_questions: Annotated[int, Field(..., description="Planned question count")]


class CollectiveInterviewTopicSchema(BaseModel):
    interview_topics: List[TopicSchema] = Field(..., description="List of interview topics")


class CollectiveInterviewTopicFeedbackSchema(BaseModel):
    satisfied: bool
    # updated_topics: CollectiveInterviewTopicSchema
    feedback: str


class DiscussionSummaryPerTopicSchema(BaseModel):
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
                "DiscussionSummaryPerTopicSchema.DeepDive"
            ]
        ]
        guidelines: str
        focus_areas_covered: List[str]
        reference_material: List[str]

    discussion_topics: List[DiscussionTopic]


QuestionType = Literal["Opening", "Direct", "Deep Dive"]


class NodeSchema(BaseModel):
    id: int = Field(..., ge=1)
    question_type: QuestionType
    question: Optional[str] = None  # Required if question_type is "Direct"
    graded: bool
    next_node: Optional[int] = Field(None, ge=1)
    context: str = Field(..., min_length=1)
    skills: List[str] = Field(..., min_items=1)
    total_question_threshold: Optional[int] = Field(None, ge=1)
    question_guidelines: Optional[str] = None


class TopicWithNodesSchema(BaseModel):
    topic: str
    nodes: List[NodeSchema] = Field(..., min_items=1)


class NodesSchema(BaseModel):
    topics_with_nodes: List[TopicWithNodesSchema] = Field(..., min_items=1)


QType = Literal["New Question", "Counter Question"]
QDiff = Literal["Easy", "Medium", "Hard"]
QCountType = Literal["Twist", "Interrogatory"]

class QAItem(BaseModel):
    qa_id: str = Field(..., description="QA identifier like 'QA1'")
    example_questions: List[str] = Field(..., min_items=5, max_items=5)


class QABlock(BaseModel):
    block_id: str = Field(..., description="Block identifier like 'B1'")
    guideline: str = Field(..., min_length=1)
    q_type: QType
    q_difficulty: QDiff
    counter_type: Optional[QCountType] = None  # required if q_type == "Counter Question"
    qa_items: List[QAItem] = Field(..., min_items=1, max_items=1)  # exactly one QA item per block


class QASet(BaseModel):
    topic: str = Field(..., min_length=1)
    qa_blocks: List[QABlock] = Field(..., min_items=7, max_items=7)  # exactly 7 blocks per topic


class QASetsSchema(BaseModel):
    qa_sets: List[QASet] = Field(..., min_items=1)


class OutputSchema(BaseModel):
    summary: GeneratedSummarySchema
    interview_topics: CollectiveInterviewTopicSchema
    discussion_summary_per_topic: DiscussionSummaryPerTopicSchema
    nodes: NodesSchema
    qa_blocks: QASetsSchema
