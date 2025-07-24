"""
Summarizer agent for creating conversation summaries using Vertex AI.
"""

from typing import Any, Dict, List

import litellm
from zenml import get_step_context

from ..utils.llm_config import (
    generate_trace_url,
    get_pipeline_run_id,
    initialize_litellm_langfuse,
)
from ..utils.manual_langfuse_logger import get_manual_langfuse_logger
from ..utils.models import ConversationData, Summary


class SummarizerAgent:
    """Agent responsible for creating conversation summaries."""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_name = (
            f"vertex_ai/{model_config.get('model_name', 'gemini-2.5-flash')}"
        )
        self.max_tokens = model_config.get("max_tokens", 4000)
        self.temperature = model_config.get("temperature", 0.1)
        self.top_p = model_config.get("top_p", 0.95)

        # Initialize LiteLLM-Langfuse integration (may fail due to version conflicts)
        self.langfuse_enabled = initialize_litellm_langfuse()

        # Initialize manual Langfuse logger as fallback
        self.manual_logger = get_manual_langfuse_logger()

    def _format_conversation_for_prompt(
        self, conversation: ConversationData
    ) -> str:
        """Format conversation data into a readable text for the LLM."""

        formatted_text = (
            f"Channel: #{conversation.channel_name} ({conversation.source})\n"
        )
        formatted_text += f"Time Range: {conversation.date_range['start']} to {conversation.date_range['end']}\n"
        formatted_text += (
            f"Participants: {conversation.participant_count} people\n\n"
        )
        formatted_text += "Messages:\n"

        for message in conversation.messages:
            timestamp = message.timestamp.strftime("%H:%M")
            formatted_text += (
                f"[{timestamp}] {message.author}: {message.content}\n"
            )

        return formatted_text

    def _get_run_id_tag(self) -> str:
        """Get ZenML run ID for tagging LLM calls."""
        try:
            step_context = get_step_context()
            return str(step_context.pipeline_run.id)
        except Exception:
            import uuid

            return str(uuid.uuid4())

    def create_summary(self, conversation: ConversationData) -> Summary:
        """Create a summary for a single conversation."""

        conversation_text = self._format_conversation_for_prompt(conversation)

        system_prompt = """You are an expert at summarizing team conversations. Create a concise, informative summary that captures the key discussion points, decisions, and outcomes.

Your summary should:
1. Be clear and professional
2. Highlight key topics and decisions
3. Identify main participants and their contributions
4. Note any important outcomes or next steps
5. Be concise but comprehensive

Format your response as follows:
TITLE: [Brief title for the conversation]
SUMMARY: [2-3 paragraph summary]
KEY_POINTS: [Bullet points of main discussion points]
PARTICIPANTS: [Key participants who contributed significantly]
TOPICS: [Main topics discussed]"""

        human_prompt = f"""Please summarize the following team conversation:

{conversation_text}

Provide a clear, professional summary following the requested format."""

        run_id = self._get_run_id_tag()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        # Get pipeline run ID for tracing
        trace_id = get_pipeline_run_id()

        # Generate timestamp for trace URL
        from datetime import datetime

        timestamp = datetime.utcnow().isoformat() + "Z"

        metadata = {
            "trace_id": trace_id,
            "generation_name": f"Summarize {conversation.channel_name} conversation ({len(conversation.messages)} messages)",
            "tags": [
                "summarizer_agent",
                "conversation_summary",
                f"channel:{conversation.channel_name}",
            ],
            "session_id": trace_id,
            "user_id": "zenml_pipeline",
            "trace_metadata": {
                "channel": conversation.channel_name,
                "source": conversation.source,
                "participant_count": conversation.participant_count,
                "message_count": len(conversation.messages),
                "trace_url": generate_trace_url(trace_id, timestamp),
            },
        }

        import time

        start_time = time.time()

        # Make LLM call without metadata if Langfuse integration failed
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            metadata=metadata if self.langfuse_enabled else {},
        )
        end_time = time.time()
        summary_text = response.choices[0].message.content

        # Manual logging if built-in integration failed
        if not False:  # self.langfuse_enabled:
            usage = getattr(response, "usage", None)
            usage_dict = None
            if usage:
                usage_dict = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            self.manual_logger.log_llm_call(
                messages=messages,
                response_content=summary_text,
                metadata=metadata,
                model=self.model_name,
                usage=usage_dict,
                start_time=start_time,
                end_time=end_time,
            )

        # Parse the response to extract structured data
        parsed_summary = self._parse_summary_response(
            summary_text, conversation
        )

        return parsed_summary

    def _parse_summary_response(
        self, response_text: str, conversation: ConversationData
    ) -> Summary:
        """Parse the LLM response into a structured Summary object."""

        lines = response_text.split("\n")

        title = ""
        content = ""
        key_points = []
        participants = []
        topics = []

        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
                current_section = "title"
            elif line.startswith("SUMMARY:"):
                content = line.replace("SUMMARY:", "").strip()
                current_section = "summary"
            elif line.startswith("KEY_POINTS:"):
                current_section = "key_points"
            elif line.startswith("PARTICIPANTS:"):
                current_section = "participants"
            elif line.startswith("TOPICS:"):
                current_section = "topics"
            elif line and current_section:
                if current_section == "summary":
                    content += " " + line
                elif current_section == "key_points" and (
                    line.startswith("-") or line.startswith("•")
                ):
                    key_points.append(line.lstrip("-•").strip())
                elif current_section == "participants":
                    # Extract participant names (assuming comma-separated)
                    if "," in line:
                        participants.extend(
                            [p.strip() for p in line.split(",")]
                        )
                    else:
                        participants.append(line)
                elif current_section == "topics":
                    # Extract topics (assuming comma-separated)
                    if "," in line:
                        topics.extend([t.strip() for t in line.split(",")])
                    else:
                        topics.append(line)

        # Fallback if parsing fails
        if not title:
            title = f"Summary of #{conversation.channel_name}"
        if not content:
            content = response_text

        # Calculate confidence score based on response quality
        confidence_score = self._calculate_confidence_score(
            response_text, conversation
        )

        return Summary(
            title=title,
            content=content.strip(),
            key_points=key_points,
            participants=participants,
            topics=topics,
            word_count=len(content.split()),
            confidence_score=confidence_score,
        )

    def _calculate_confidence_score(
        self, summary_text: str, conversation: ConversationData
    ) -> float:
        """Calculate a confidence score for the summary quality."""

        score = 0.5  # Base score

        # Check if summary has reasonable length
        word_count = len(summary_text.split())
        if 50 <= word_count <= 500:
            score += 0.2

        # Check if key participants are mentioned
        conversation_participants = set(
            msg.author for msg in conversation.messages
        )
        mentioned_participants = sum(
            1
            for participant in conversation_participants
            if participant.lower() in summary_text.lower()
        )
        if mentioned_participants >= len(conversation_participants) * 0.5:
            score += 0.2

        # Check if summary follows expected structure
        if "TITLE:" in summary_text and "SUMMARY:" in summary_text:
            score += 0.1

        return min(score, 1.0)

    def create_multi_conversation_summary(
        self, conversations: List[ConversationData]
    ) -> Summary:
        """Create a combined summary for multiple conversations."""

        if len(conversations) == 1:
            return self.create_summary(conversations[0])

        # Create individual summaries first
        individual_summaries = []
        for conversation in conversations:
            summary = self.create_summary(conversation)
            individual_summaries.append(summary)

        # Combine summaries
        combined_text = "\n\n".join(
            [
                f"#{conv.channel_name} ({conv.source}): {summary.content}"
                for conv, summary in zip(conversations, individual_summaries)
            ]
        )

        system_prompt = """You are creating a daily digest from multiple team conversations. Combine the individual channel summaries into a cohesive daily overview that highlights cross-channel themes, important decisions, and key outcomes.

Your combined summary should:
1. Identify common themes across channels
2. Highlight the most important decisions and outcomes
3. Note any cross-team collaboration or dependencies
4. Provide a clear daily overview for team members"""

        human_prompt = f"""Please create a daily digest from these channel summaries:

{combined_text}

Provide a comprehensive daily overview that synthesizes the key information."""

        run_id = self._get_run_id_tag()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        metadata = {
            "tags": [
                f"run_id:{run_id}",
                "summarizer_agent",
                "multi_conversation_summary",
            ],
            "trace_name": "daily-digest",
            "session_id": run_id,
            "trace_user_id": "zenml_pipeline",
            "generation_name": "daily_digest",
            "trace_metadata": {
                "total_conversations": len(conversations),
                "channels": [c.channel_name for c in conversations],
                "sources": list(set(c.source for c in conversations)),
            },
        }

        import time

        start_time = time.time()

        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            metadata=metadata if self.langfuse_enabled else {},
        )

        end_time = time.time()

        # Manual logging if built-in integration failed
        if not self.langfuse_enabled:
            usage = getattr(response, "usage", None)
            usage_dict = None
            if usage:
                usage_dict = {
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            self.manual_logger.log_llm_call(
                messages=messages,
                response_content=response.choices[0].message.content,
                metadata=metadata,
                model=self.model_name,
                usage=usage_dict,
                start_time=start_time,
                end_time=end_time,
            )

        # Combine metadata from all conversations
        all_participants = set()
        all_topics = []
        for summary in individual_summaries:
            all_participants.update(summary.participants)
            all_topics.extend(summary.topics)

        return Summary(
            title="Daily Team Digest",
            content=response.choices[0].message.content,
            key_points=[],  # Could extract from combined summary
            participants=list(all_participants),
            topics=list(set(all_topics)),  # Remove duplicates
            word_count=len(response.choices[0].message.content.split()),
            confidence_score=0.8,  # Default confidence for combined summaries
        )
