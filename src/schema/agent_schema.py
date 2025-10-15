from typing import Annotated

from langchain_core.messages import AnyMessage, RemoveMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, ConfigDict

from .input_schema import (
    CandidateProfileSchema,
    JobDescriptionSchema,
    QuestionGuidelinesCompleteSchema,
    SkillTreeSchema,
)
from .output_schema import (
    CollectiveInterviewTopicFeedbackSchema,
    CollectiveInterviewTopicSchema,
    DiscussionSummaryPerTopicSchema,
    GeneratedSummarySchema,
    NodesSchema,
    QASetsSchema,
)


class AgentInternalState(BaseModel):
    """
    Central state object threaded through the agenda-generation pipeline.

    This model carries:
      • Mongo connection details (strings only; drivers are created elsewhere)
      • Conversation/thread id (used as a stable key for persistence)
      • Running message list (consumed by LangGraph via `add_messages`)
      • All inputs (JD, skill tree, candidate profile, guidelines)
      • All intermediate/terminal artifacts (summary, topics, per-topic summaries,
        nodes, QA blocks) and corresponding error strings/feedback, when present.
    """

    # Allow usage of non-Pydantic (arbitrary) types such as LangChain message objects.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -------------------- Persistence / runtime identifiers --------------------
    mongo_client: str  # Mongo connection URI
    mongo_db: str  # Database name
    mongo_jd_collection: str  # Collection for Job Descriptions
    mongo_cv_collection: str  # Collection for Candidate Profiles
    mongo_skill_tree_collection: str  # Collection for Skill Trees
    mongo_summary_collection: str  # Collection for Generated Summaries
    mongo_question_guidelines_collection: str  # Collection for Question Guidelines
    id: str  # Thread/session identifier used as a stable _id across documents

    # Running message buffer used by LangGraph; `add_messages` appends new messages.
    messages: Annotated[list[AnyMessage | RemoveMessage], add_messages] = []

    # -------------------- Inputs --------------------
    job_description: JobDescriptionSchema
    skill_tree: SkillTreeSchema
    candidate_profile: CandidateProfileSchema
    question_guidelines: QuestionGuidelinesCompleteSchema

    # -------------------- Outputs / intermediates --------------------
    generated_summary: GeneratedSummarySchema | None = None
    interview_topics: CollectiveInterviewTopicSchema | None = None
    discussion_summary_per_topic: DiscussionSummaryPerTopicSchema | None = None
    nodes: NodesSchema | None = None
    qa_blocks: QASetsSchema | None = None

    # -------------------- Error strings / feedback channels --------------------
    nodes_error: str = ""  # Accumulates node-generation validation or runtime issues
    qa_error: str = ""  # Accumulates QA-block generation validation or runtime issues
    interview_topics_feedback: CollectiveInterviewTopicFeedbackSchema | None = None
    # Appending text log of feedback prompts given to the topic generator across retries
    interview_topics_feedbacks: str = ""
