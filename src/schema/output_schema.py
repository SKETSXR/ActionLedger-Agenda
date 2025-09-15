import re
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Optional, Dict, List, Union, Literal, Tuple

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


class CandidateProjectSummarySchema(BaseModel):
    projectwise_summary: Annotated[
        list[str],
        Field(...,
            description="Project-wise Summaries",
            examples=["AI Applications Engineer at Jewelers Mutual Group: Developed generative AI applications using LLMs from Azure OpenAI to enhance automation and decision-making workflows. Integrated AI-driven document processing solutions using LangChain, Python, and cloud-native architectures.", 
                      "AI Engineer at PeritusHub (Stealth Start-Up): Implemented a scalable multi-agent system using Langchain and AWS Bedrock for conversational AI tasks. Managed CI/CD pipeline using GitHub Actions and engineered a Document Management System incorporating RAG with OpenAI LLMs.",
                      "Data Scientist Intern at H-E-B Groceries: Developed a system for extracting product entities and attributes from images using Azure OpenAI LLM and computer vision techniques. Created a web-based user interface for the extraction system using HTML, JavaScript, and Flask."]
        )
    ]


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


class TopicSchema(BaseModel):
    topic: Annotated[str, Field(..., description="Short name of the discussion topic")]
    why_this_topic: Annotated[str, Field(..., description="A short reason for why this discussion topic has been chosen")]
    focus_area: Annotated[Dict[str, str], Field(..., description="skill -> focus description")]
    necessary_reference_material: Annotated[str, Field(..., description="Reference material for this topic")]
    total_questions: Annotated[int, Field(..., description="Planned question count")]


class CollectiveInterviewTopicSchema(BaseModel):
    interview_topics: List[TopicSchema] = Field(..., description="List of interview topics")


# class DiscussionSummaryPerTopicSchema(BaseModel):
#     listoftopics_discussion_summary: list = Field(..., description="Topic name")
#     summary: list[str] = Field(..., description="Summary of the discussion for this topic")


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


# BLOCK_ID_RE = re.compile(r"^B[1-9]\d*$")   # B1, B2, ...
# QA_ID_RE    = re.compile(r"^QA[1-9]\d*$")  # QA1, QA2, ...


# class QABlock(BaseModel):
#     block_id: str = Field(..., description="Block identifier like 'B1'")
#     qa_id: str = Field(..., description="QA identifier like 'QA1'")
#     guideline: str = Field(..., min_length=1)
#     q_type: Literal["First Question", "New Question", "Counter Question"]
#     q_difficulty: Literal["Easy", "Medium", "Hard"]
#     example_questions: List[str] = Field(
#         ..., min_items=1,
#         description="A set of deep dive QA sample questions",
#         examples=[
#             "Can you describe a project where you applied prompt engineering to improve LLM outputs, and what specific techniques did you use?",
#             "What challenges did you face when fine-tuning LLM outputs using prompt engineering, and how did you overcome them?",
#             "How do you determine the effectiveness of prompt engineering techniques in enhancing LLM performance?",
#             "In your experience, what are the key factors to consider when applying prompt engineering to LLMs for specific tasks?",
#             "Can you provide an example of how you optimized a prompt to achieve better results in an LLM application?"
#           ]
#     )


# class QASet(BaseModel):
#     topic: str = Field(..., min_length=1, description="Readable topic name")
#     qa_blocks: List[QABlock] = Field(..., min_items=1)


# class QASetsSchema(BaseModel):
#     qa_sets: List[QASet] = Field(..., min_items=1)


# QType = Literal["First Question", "New Question", "Counter Question"]
# QDiff = Literal["Easy", "Medium", "Hard"]

# class QAItem(BaseModel):
#     qa_id: str = Field(..., description="QA identifier like 'QA1'")
#     q_type: QType
#     q_difficulty: QDiff
#     example_questions: List[str] = Field(..., min_items=5, max_items=5)

# class QABlock(BaseModel):
#     block_id: str = Field(..., description="Block identifier like 'B1'")
#     guideline: str = Field(..., min_length=1)
#     qa_items: List[QAItem] = Field(..., min_items=8, max_items=8)


# class QASet(BaseModel):
#     topic: str = Field(..., min_length=1)
#     qa_blocks: List[QABlock] = Field(..., min_items=1)


# class QASetsSchema(BaseModel):
#     qa_sets: List[QASet] = Field(..., min_items=1)


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
