
import json
import asyncio
from typing import List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from src.mongo_tools import get_mongo_tools
from ..schema.agent_schema import AgentInternalState
from ..schema.output_schema import DiscussionSummaryPerTopicSchema
from ..prompt.discussion_summary_per_topic_generation_agent_prompt import (
    DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT
)
from ..model_handling import llm_dts

# set_llm_cache(InMemoryCache())


class PerTopicDiscussionSummaryGenerationAgent:        
    llm_dts = llm_dts
    MONGO_TOOLS = get_mongo_tools(llm=llm_dts)
    llm_dts_with_tools = llm_dts.bind_tools(MONGO_TOOLS) 

    @staticmethod
    async def _one_topic_call(generated_summary_json: str, topic: Dict[str, Any], thread_id):
        """Call LLM once for a single topic and return a structured DiscussionTopic."""
        TopicEntry = DiscussionSummaryPerTopicSchema.DiscussionTopic
        resp = await PerTopicDiscussionSummaryGenerationAgent.llm_dts_with_tools \
            .with_structured_output(TopicEntry, method="function_calling") \
            .ainvoke([
                SystemMessage(
                    content=DISCUSSION_SUMMARY_PER_TOPIC_GENERATION_AGENT_PROMPT.format(
                        generated_summary=generated_summary_json,
                        thread_id=thread_id,
                        interview_topic=json.dumps(topic)
                    )
                )
            ])
        return resp

    @staticmethod
    async def should_regenerate(state: AgentInternalState):
        input_topics = {t.topic for t in state.interview_topics.interview_topics}
        output_topics = {dt.topic for dt in state.discussion_summary_per_topic.discussion_topics}
        if input_topics != output_topics:
            missing = input_topics - output_topics
            extra = output_topics - input_topics
            print(f"[PerTopic] Topic mismatch: missing {missing}, extra {extra}")
            return True
        else:
            return False

    @staticmethod
    async def discussion_summary_per_topic_generator(state: AgentInternalState) -> AgentInternalState:
        """
        Parallel per-topic summaries using asyncio.gather (no inner subgraph).
        Produces DiscussionSummaryPerTopicSchema with one entry per input topic.
        """
        # Normalize topics list coming from parent state
        try:
            topics_list: List[Dict[str, Any]] = [t.model_dump() for t in state.interview_topics.interview_topics]
        except Exception:
            topics_list = state.interview_topics  # already a list[dict]

        if not isinstance(topics_list, list) or len(topics_list) == 0:
            raise ValueError("interview_topics must be a non-empty list[dict]")

        # Serialize generated_summary for prompt
        try:
            generated_summary_json = state.generated_summary.model_dump_json()
        except Exception:
            generated_summary_json = json.dumps(state.generated_summary)

        # Run all topic calls concurrently
        tasks = [
            asyncio.create_task(PerTopicDiscussionSummaryGenerationAgent._one_topic_call(generated_summary_json, topic, state.id))
            for topic in topics_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful DiscussionTopic entries
        discussion_topics = []
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                continue
            # r is already a structured DiscussionSummaryPerTopicSchema.DiscussionTopic
            try:
                discussion_topics.append(r)
            except Exception as e:
                print("Topic %d: could not append structured response (%s)", idx, e)

        state.discussion_summary_per_topic = DiscussionSummaryPerTopicSchema(
            discussion_topics=discussion_topics
        )
        return state

    @staticmethod
    def get_graph(checkpointer=None):
        gb = StateGraph(AgentInternalState)
        gb.add_node(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.discussion_summary_per_topic_generator
        )
        gb.add_edge(START, "discussion_summary_per_topic_generator")
        gb.add_conditional_edges(
            "discussion_summary_per_topic_generator",
            PerTopicDiscussionSummaryGenerationAgent.should_regenerate,
            {
                False: END,
                True: "discussion_summary_per_topic_generator"
            }
        )
        gb.add_edge("discussion_summary_per_topic_generator", END)
        return gb.compile(checkpointer=checkpointer, name="AgendaGenerationAgent")


if __name__ == "__main__":
    graph = PerTopicDiscussionSummaryGenerationAgent.get_graph()
    print(graph.get_graph().draw_ascii())
