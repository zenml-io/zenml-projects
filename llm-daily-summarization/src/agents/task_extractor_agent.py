"""
Task extractor agent for identifying action items and tasks from conversations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import litellm
from zenml import get_step_context

from ..prompts import TASK_EXTRACTOR_HUMAN_PROMPT, TASK_EXTRACTOR_SYSTEM_PROMPT
from ..utils.llm_config import (
    generate_trace_url,
    get_pipeline_run_id,
    initialize_litellm_langfuse,
)
from ..utils.models import ConversationData, TaskItem


class TaskExtractorAgent:
    """Agent responsible for extracting tasks and action items from conversations."""

    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.model_name = (
            f"vertex_ai/{model_config.get('model_name', 'gemini-2.5-flash')}"
        )
        self.max_tokens = model_config.get("max_tokens", 2000)
        self.temperature = model_config.get("temperature", 0.1)
        self.top_p = model_config.get("top_p", 0.95)

        # Initialize LiteLLM-Langfuse integration (may fail due to version conflicts)
        self.langfuse_enabled = initialize_litellm_langfuse()

        # Task indication keywords
        self.task_indicators = [
            "todo",
            "task",
            "action item",
            "follow up",
            "need to",
            "should",
            "will do",
            "can you",
            "please",
            "assign",
            "deadline",
            "due",
            "by when",
            "schedule",
            "plan",
            "implement",
            "create",
            "build",
            "fix",
            "review",
            "update",
            "check",
            "test",
            "deploy",
        ]

    def _format_conversation_for_prompt(
        self, conversation: ConversationData
    ) -> str:
        """Format conversation data focusing on actionable content."""

        formatted_text = (
            f"Channel: #{conversation.channel_name} ({conversation.source})\n"
        )
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

    def extract_tasks(self, conversation: ConversationData) -> List[TaskItem]:
        """Extract tasks and action items from a conversation."""

        conversation_text = self._format_conversation_for_prompt(conversation)

        system_prompt = TASK_EXTRACTOR_SYSTEM_PROMPT
        human_prompt = TASK_EXTRACTOR_HUMAN_PROMPT.format(
            conversation_text=conversation_text
        )

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
            "generation_name": f"Extract tasks from {conversation.channel_name} ({len(conversation.messages)} messages)",
            "tags": [
                "task_extractor_agent",
                "task_extraction",
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

        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            metadata=metadata if self.langfuse_enabled else {},
        )
        response_content = response.choices[0].message.content

        tasks = self._parse_task_response(response_content, conversation)

        return tasks

    def _parse_task_response(
        self, response_text: str, conversation: ConversationData
    ) -> List[TaskItem]:
        """Parse the LLM response into TaskItem objects."""

        tasks = []
        lines = response_text.split("\n")

        current_task = {}
        parsing_task = False

        for line in lines:
            line = line.strip()

            if line == "TASK_START":
                parsing_task = True
                current_task = {}
            elif line == "TASK_END":
                if parsing_task and current_task:
                    task = self._create_task_item(current_task, conversation)
                    if task:
                        tasks.append(task)
                    current_task = {}
                parsing_task = False
            elif parsing_task:
                if line.startswith("TITLE:"):
                    current_task["title"] = line.replace("TITLE:", "").strip()
                elif line.startswith("DESCRIPTION:"):
                    current_task["description"] = line.replace(
                        "DESCRIPTION:", ""
                    ).strip()
                elif line.startswith("ASSIGNEE:"):
                    current_task["assignee"] = line.replace(
                        "ASSIGNEE:", ""
                    ).strip()
                elif line.startswith("PRIORITY:"):
                    current_task["priority"] = line.replace(
                        "PRIORITY:", ""
                    ).strip()
                elif line.startswith("DUE_DATE:"):
                    current_task["due_date"] = line.replace(
                        "DUE_DATE:", ""
                    ).strip()
                elif line.startswith("SOURCE_MESSAGES:"):
                    current_task["source_messages"] = line.replace(
                        "SOURCE_MESSAGES:", ""
                    ).strip()
                elif line.startswith("CONFIDENCE:"):
                    current_task["confidence"] = line.replace(
                        "CONFIDENCE:", ""
                    ).strip()

        return tasks

    def _create_task_item(
        self, task_data: Dict[str, str], conversation: ConversationData
    ) -> TaskItem:
        """Create a TaskItem from parsed data."""

        try:
            # Parse due date
            due_date = None
            if task_data.get("due_date") and task_data[
                "due_date"
            ].lower() not in ["none", "unspecified"]:
                try:
                    due_date = datetime.strptime(
                        task_data["due_date"], "%Y-%m-%d"
                    )
                except ValueError:
                    # Try to parse relative dates
                    due_date = self._parse_relative_date(task_data["due_date"])

            # Parse assignee
            assignee = task_data.get("assignee", "").strip()
            if assignee.lower() in ["unassigned", "none", ""]:
                assignee = None

            # Parse confidence score
            confidence_score = 0.5
            try:
                confidence_score = float(task_data.get("confidence", "0.5"))
            except ValueError:
                pass

            # Get source message IDs (map author names to message IDs)
            source_messages = []
            source_authors = task_data.get("source_messages", "").split(",")
            for message in conversation.messages:
                if any(
                    author.strip().lower() in message.author.lower()
                    for author in source_authors
                ):
                    source_messages.append(message.id)

            return TaskItem(
                title=task_data.get("title", "Untitled Task"),
                description=task_data.get("description", ""),
                assignee=assignee,
                priority=task_data.get("priority", "medium").lower(),
                due_date=due_date,
                source_messages=source_messages,
                confidence_score=confidence_score,
            )

        except Exception as e:
            # Log error but don't fail the entire extraction
            print(f"Error creating task item: {e}")
            return None

    def _parse_relative_date(self, date_text: str) -> datetime:
        """Parse relative date expressions."""

        now = datetime.now()
        date_text = date_text.lower()

        if "today" in date_text:
            return now
        elif "tomorrow" in date_text:
            return now + timedelta(days=1)
        elif "next week" in date_text:
            return now + timedelta(weeks=1)
        elif "friday" in date_text:
            # Find next Friday
            days_ahead = 4 - now.weekday()  # Friday is 4
            if days_ahead <= 0:
                days_ahead += 7
            return now + timedelta(days=days_ahead)
        elif "monday" in date_text:
            # Find next Monday
            days_ahead = 0 - now.weekday()  # Monday is 0
            if days_ahead <= 0:
                days_ahead += 7
            return now + timedelta(days=days_ahead)

        return None

    def extract_tasks_from_multiple_conversations(
        self, conversations: List[ConversationData]
    ) -> List[TaskItem]:
        """Extract tasks from multiple conversations."""

        all_tasks = []

        for conversation in conversations:
            conversation_tasks = self.extract_tasks(conversation)
            all_tasks.extend(conversation_tasks)

        # Deduplicate similar tasks
        deduplicated_tasks = self._deduplicate_tasks(all_tasks)

        return deduplicated_tasks

    def _deduplicate_tasks(self, tasks: List[TaskItem]) -> List[TaskItem]:
        """Remove duplicate or very similar tasks."""

        if len(tasks) <= 1:
            return tasks

        unique_tasks = []

        for task in tasks:
            is_duplicate = False

            for existing_task in unique_tasks:
                # Check if tasks are very similar
                title_similarity = self._calculate_similarity(
                    task.title.lower(), existing_task.title.lower()
                )
                desc_similarity = self._calculate_similarity(
                    task.description.lower(), existing_task.description.lower()
                )

                if title_similarity > 0.8 or desc_similarity > 0.9:
                    # Merge tasks, keeping the one with higher confidence
                    if task.confidence_score > existing_task.confidence_score:
                        unique_tasks.remove(existing_task)
                        unique_tasks.append(task)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_tasks.append(task)

        return unique_tasks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words."""

        if not text1 or not text2:
            return 0.0

        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)
