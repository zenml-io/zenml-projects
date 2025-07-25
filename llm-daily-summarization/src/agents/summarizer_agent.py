"""
Summarizer agent for creating conversation summaries using Vertex AI.
"""

import concurrent.futures
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import litellm
from zenml import get_step_context

from ..prompts import (
    DAILY_DIGEST_HUMAN_PROMPT,
    DAILY_DIGEST_SYSTEM_PROMPT,
    SUMMARIZER_HUMAN_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
)
from ..utils.llm_config import (
    generate_trace_url,
    get_pipeline_run_id,
    initialize_litellm_langfuse,
)
from ..utils.models import ConversationData, Summary


class SummarizerAgent:
    """Agent responsible for creating conversation summaries."""

    def __init__(self, model_config: Dict[str, Any], max_workers: int = 4):
        self.model_config = model_config
        self.model_name = (
            f"vertex_ai/{model_config.get('model_name', 'gemini-2.5-flash')}"
        )
        self.max_tokens = model_config.get("max_tokens", 4000)
        self.temperature = model_config.get("temperature", 0.1)
        self.top_p = model_config.get("top_p", 0.95)

        # Parallelism / rate-limiting parameters
        self.max_workers = model_config.get("max_workers", max_workers)
        # Semaphore ensures we never have more than max_workers LLM calls in-flight
        self._semaphore = threading.Semaphore(self.max_workers)

        # Initialize LiteLLM-Langfuse integration (may fail due to version conflicts)
        self.langfuse_enabled = initialize_litellm_langfuse()

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
            return str(uuid.uuid4())

    def create_summary(self, conversation: ConversationData) -> Summary:
        """Create a summary for a single conversation."""

        conversation_text = self._format_conversation_for_prompt(conversation)

        system_prompt = SUMMARIZER_SYSTEM_PROMPT
        human_prompt = SUMMARIZER_HUMAN_PROMPT.format(
            conversation_text=conversation_text
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]

        # Get pipeline run ID for tracing
        trace_id = get_pipeline_run_id()

        # Generate timestamp for trace URL
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

        # Make LLM call with metadata
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            metadata=metadata if self.langfuse_enabled else {},
        )
        summary_text = response.choices[0].message.content

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

    def _safe_create_summary(
        self, conversation: ConversationData
    ) -> Tuple[Summary | None, str | None]:
        """Thread-safe wrapper around create_summary that captures errors.

        Returns:
            (Summary | None, str | None): Tuple where the first element is the
            generated Summary (or None if failed) and the second element is an
            error message (or None if successful).
        """
        # Acquire the semaphore to respect the concurrency cap
        self._semaphore.acquire()
        try:
            summary = self.create_summary(conversation)
            return summary, None
        except Exception as e:
            return None, f"Error summarizing {conversation.channel_name}: {e}"
        finally:
            # Always release so other threads can proceed
            self._semaphore.release()

    def create_multi_conversation_summary(
        self, conversations: List[ConversationData]
    ) -> Tuple[Summary, List[str]]:
        """Create a combined summary for multiple conversations (parallel).

        NOTE: Method now returns a tuple (combined_summary, errors). Down-stream
        callers must handle the additional error list.
        """

        # Single conversation shortcut (maintain existing behaviour)
        if len(conversations) == 1:
            summary = self.create_summary(conversations[0])
            return summary, []

        # Parallel summarization of individual conversations
        individual_summaries: List[Summary] = []
        errors: List[str] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Map conversations to _safe_create_summary
            futures = {
                executor.submit(self._safe_create_summary, conv): conv
                for conv in conversations
            }

            for future in concurrent.futures.as_completed(futures):
                summary, error = future.result()
                if summary:
                    individual_summaries.append(summary)
                if error:
                    errors.append(error)

        # If no successful summaries, we cannot build a combined summary
        if not individual_summaries:
            raise RuntimeError(
                "Failed to create any conversation summaries. "
                f"Errors: {errors}"
            )

        # Combine successful summaries
        combined_text_parts = []
        for conv, summary in zip(conversations, individual_summaries):
            combined_text_parts.append(
                f"#{conv.channel_name} ({conv.source}): {summary.content}"
            )
        combined_text = "\n\n".join(combined_text_parts)

        system_prompt = DAILY_DIGEST_SYSTEM_PROMPT
        human_prompt = DAILY_DIGEST_HUMAN_PROMPT.format(
            combined_text=combined_text
        )

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
                "total_conversations": len(individual_summaries),
                "channels": [c.channel_name for c in conversations],
                "sources": list(set(c.source for c in conversations)),
            },
        }

        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            metadata=metadata if self.langfuse_enabled else {},
        )

        # Combine metadata from all individual summaries
        all_participants = set()
        all_topics = []
        for summary in individual_summaries:
            all_participants.update(summary.participants)
            all_topics.extend(summary.topics)

        combined_summary = Summary(
            title="Daily Team Digest",
            content=response.choices[0].message.content,
            key_points=[],  # Could extract from combined summary
            participants=list(all_participants),
            topics=list(set(all_topics)),  # Remove duplicates
            word_count=len(response.choices[0].message.content.split()),
            confidence_score=0.8,  # Default confidence for combined summaries
        )

        # Return both combined summary and list of individual errors
        return combined_summary, errors
