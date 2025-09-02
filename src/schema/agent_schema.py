from typing import Annotated
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, RemoveMessage
from langgraph.graph import add_messages

from .output_schema import GeneratedSummarySchema, CollectiveInterviewTopicSchema
from .input_schema import JobDescriptionSchema, SkillTreeSchema, CandidateProfileSchema


class AgentInternalState(BaseModel):

    class Config:
        arbitrary_types_allowed=True

    mongo_client: str
    mongo_db: str
    # mongo_relevant_evidence_collection: str
    mongo_jd_collection: str
    mongo_cv_collection: str
    mongo_skill_tree_collection: str
    mongo_summary_collection: str
    # mongo_inferred_topics_collection: str
    id: str

    messages: Annotated[list[AnyMessage | RemoveMessage], add_messages] = []

    job_description: JobDescriptionSchema
    skill_tree: SkillTreeSchema
    candidate_profile: CandidateProfileSchema

    generated_summary: GeneratedSummarySchema | None = None
    interview_topics: CollectiveInterviewTopicSchema | None = None
    # navigation_decision: NavigationSchema | None = None
