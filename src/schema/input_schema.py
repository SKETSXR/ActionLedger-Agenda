from pydantic import BaseModel, Field
from typing import Annotated, Optional


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
    children: Optional[list['SkillTreeSchema']]
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


class InputSchema(BaseModel):
    job_description: JobDescriptionSchema
    skill_tree: SkillTreeSchema
    candidate_profile: CandidateProfileSchema
