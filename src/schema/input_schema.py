from pydantic import BaseModel, Field
from typing import Annotated, Optional, List, Literal


class ExperienceProjectBaseSchema(BaseModel):
    title: Annotated[
        str,
        Field(..., description="Title of the project or experience", examples=["Realtime Chat App"])
    ]
    description: Annotated[
        str | None,
        Field(...,
            description="Brief summary of the project or experience",
            examples=["Built a scalable chat system using WebSockets and Redis"]
        )
    ]


class ProjectSchema(ExperienceProjectBaseSchema):
    id: Annotated[
        str,
        Field(...,
            description="A unique id associated to each project",
            examples=["P1", "P2", "P3"]
        )
    ]


class ExperienceSchema(ExperienceProjectBaseSchema):
    id: Annotated[
        str,
        Field(...,
            description="A unique id associated to each experience",
            examples=["E1", "E2", "E3"]
        )
    ]
    company: Annotated[
        str,
        Field(...,
            description="Company where the candidate gained the experience",
            examples=["OpenAI"]
        )
    ]


class CandidateProfileSchema(BaseModel):
    skills: Annotated[
        list[str],
        Field(...,
            description="List of relevant skills declared by the candidate",
            examples=["Python", "Django", "SQL"]
        )
    ]
    projects: Annotated[
        list[ProjectSchema] | None,
        Field(description="List of relevant projects the candidate has completed", default=None)
    ]
    experience: Annotated[
        list[ExperienceSchema] | None,
        Field(description="List of professional work experiences", default=None)
    ]


class SkillTreeSchema(BaseModel):
    children: Optional[list['SkillTreeSchema']] = None
    name: Annotated[
        str,
        Field(...,
            description="Name of the skill or domain",
            examples=["Databases"]
        )
    ]
    weight: Annotated[
        float,
        Field(...,
            description="Importance of this skill/domain in the overall evaluation. Value is normalized between 0 and 1", examples=[0.2]
        )
    ]
    priority: Literal["must", "high", "low"] = Field(..., description="Priority of the skill or domain", examples=["must", "high", "low"])


class JobDescriptionSchema(BaseModel):
    job_role: Annotated[
        str,
        Field(...,
            description="Brief summary of the job role and required competencies",
            examples=["Backend Developer with experience in Django and REST APIs"]
        )
    ]
    company_background: Annotated[
        str, Field(...,
            description="Overview of the company's mission, projects, or focus area",
            examples=["We build AI-driven products for enterprise automation"]
        )
    ]
    fundamental_knowledge: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Educational qualifications if required",
            examples=["B.Tech or M.Tech in any field"]
        )
    ]
    # cannot_skip_skills: Annotated[
    #     list[str],
    #     Field(
    #         description="Mandatory skills",
    #         examples=["Node.js (advanced services, concurrency, resilience)", "Python (AI systems integration, data flows)", "TypeScript (advanced)"]
    #     )
    # ]
    # optional_topics_high_priority: Annotated[
    #     Optional[list[str]],
    #     Field(
    #         default=None,
    #         description="Optional Topics/Skills but having a High Priority",
    #         examples=["TypeScript full-stack proficiency", "AWS ECS/ECR scaling, autoscaling policies"]
    #     )
    # ]
    # optional_topics_low_priority: Annotated[
    #     Optional[list[str]],
    #     Field(
    #         default=None,
    #         description="Optional Topics/Skills but having a Low Priority",
    #         examples=["Kubernetes (EKS) & service mesh (Istio/Linkerd)", "gRPC & streaming APIs"]
    #     )
    # ]


class QuestionGuidelinesSchema(BaseModel):
    question_type_name: Annotated[
        str, Field(..., description="Question type to be used for guidelines and also used as a unique identifier for each mongo db record", min_length=1)
    ]
    question_guidelines: Annotated[
        str, Field(..., description="Guidelines for example question generations", min_length=1)
    ]


class QuestionGuidelinesCompleteSchema(BaseModel):
    question_guidelines: Annotated[
        List[QuestionGuidelinesSchema],
        Field(..., description="List of question guidelines to be used"),
    ]


class InputSchema(BaseModel):
    job_description: JobDescriptionSchema
    skill_tree: SkillTreeSchema
    candidate_profile: CandidateProfileSchema
    question_guidelines: QuestionGuidelinesCompleteSchema
