from typing import Annotated, Optional, List, Literal
from pydantic import BaseModel, Field


# ------------------------------ Candidate profile ------------------------------ #
class ExperienceProjectBaseSchema(BaseModel):
    """
    Common base for a candidate's projects and experiences.
    Contains a title and a short description.
    """

    title: Annotated[
        str,
        Field(
            ...,
            description="Title of the project or experience",
            examples=["Realtime Chat App"],
        ),
    ]
    description: Annotated[
        str | None,
        Field(
            ...,
            description="Brief summary of the project or experience",
            examples=["Built a scalable chat system using WebSockets and Redis"],
        ),
    ]


class ProjectSchema(ExperienceProjectBaseSchema):
    """A single project entry."""

    id: Annotated[
        str,
        Field(
            ...,
            description="A unique id associated to each project",
            examples=["P1", "P2", "P3"],
        ),
    ]


class ExperienceSchema(ExperienceProjectBaseSchema):
    """A single professional experience entry."""

    id: Annotated[
        str,
        Field(
            ...,
            description="A unique id associated to each experience",
            examples=["E1", "E2", "E3"],
        ),
    ]
    company: Annotated[
        str,
        Field(
            ...,
            description="Company where the candidate gained the experience",
            examples=["OpenAI"],
        ),
    ]


class CandidateProfileSchema(BaseModel):
    """The candidate profile having: skills, projects, and work experience."""

    skills: Annotated[
        list[str],
        Field(
            ...,
            description="List of relevant skills declared by the candidate",
            examples=["Python", "Django", "SQL"],
        ),
    ]
    projects: Annotated[
        list[ProjectSchema] | None,
        Field(
            default=None,
            description="List of relevant projects the candidate has completed",
        ),
    ]
    experience: Annotated[
        list[ExperienceSchema] | None,
        Field(default=None, description="List of professional work experiences"),
    ]


# --------------------------------- Skill tree --------------------------------- #
class SkillTreeSchema(BaseModel):
    """
    Recursive skill tree representing domains/skills and their relative weights
    and priority. Leaf nodes have `children=None`.
    """

    children: Optional[list["SkillTreeSchema"]] = None
    name: Annotated[
        str,
        Field(
            ...,
            description="Name of the skill or domain",
            examples=["Databases"],
        ),
    ]
    weight: Annotated[
        float,
        Field(
            ...,
            description=(
                "Importance of this skill/domain in the overall evaluation. "
                "Value is normalized between 0 and 1"
            ),
            examples=[0.2],
        ),
    ]
    priority: Literal["must", "high", "low"] = Field(
        ...,
        description="Priority of the skill or domain",
        examples=["must", "high", "low"],
    )


# ------------------------------ Job description ------------------------------- #
class JobDescriptionSchema(BaseModel):
    """Job role context and basic requirements for the position."""

    job_role: Annotated[
        str,
        Field(
            ...,
            description="Brief summary of the job role and required competencies",
            examples=["Backend Developer with experience in Django and REST APIs"],
        ),
    ]
    company_background: Annotated[
        str,
        Field(
            ...,
            description="Overview of the company's mission, projects, or focus area",
            examples=["We build AI-driven products for enterprise automation"],
        ),
    ]
    fundamental_knowledge: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Educational qualifications if required",
            examples=["B.Tech or M.Tech in any field"],
        ),
    ]


# ---------------------------- Question guidelines ----------------------------- #
class QuestionGuidelinesSchema(BaseModel):
    """Guideline text attached to a specific question type (unique identifier)."""

    question_type_name: Annotated[
        str,
        Field(
            ...,
            description=(
                "Question type to be used for guidelines and also used as a unique "
                "identifier for each mongo db record"
            ),
            min_length=1,
        ),
    ]
    question_guidelines: Annotated[
        str,
        Field(
            ...,
            description="Guidelines for example question generations",
            min_length=1,
        ),
    ]


class QuestionGuidelinesCompleteSchema(BaseModel):
    """Container for all question guidelines."""

    question_guidelines: Annotated[
        List[QuestionGuidelinesSchema],
        Field(..., description="List of question guidelines to be used"),
    ]


# --------------------------------- Input root --------------------------------- #
class InputSchema(BaseModel):
    """
    Root input payload for the agenda generation pipeline.
    Includes the JD, skill tree, candidate profile, and question guidelines.
    """

    job_description: JobDescriptionSchema
    skill_tree: SkillTreeSchema
    candidate_profile: CandidateProfileSchema
    question_guidelines: QuestionGuidelinesCompleteSchema
