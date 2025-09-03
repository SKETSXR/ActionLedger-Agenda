from pydantic import BaseModel, Field
from typing import Annotated, Optional, Dict, List

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
    annotated_skill_tree: AnnotatedSkillTreeSummarySchema
    domains_assess: DomainsToAssessListSchema
    # total_questions: TotalQuestionsSchema


class TopicSchema(BaseModel):
    topic: Annotated[str, Field(..., description="Short name of the discussion topic")]
    focus_area: Annotated[Dict[str, str], Field(..., description="skill -> focus description")]
    necessary_reference_material: Annotated[str, Field(..., description="Reference material for this topic")]
    total_questions: Annotated[int, Field(..., description="Planned question count")]


class CollectiveInterviewTopicSchema(BaseModel):
    interview_topics: List[TopicSchema] = Field(..., description="List of interview topics")


class OutputSchema(BaseModel):
    summary: GeneratedSummarySchema
    interview_topics: CollectiveInterviewTopicSchema
