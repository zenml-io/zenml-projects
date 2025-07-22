"""
Task extractor agent for identifying action items and tasks from conversations.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta

from langchain_google_vertexai import ChatVertexAI
from langchain.schema import HumanMessage, SystemMessage
from langfuse import observe

from ..utils.models import ConversationData, TaskItem
from ..utils.session_manager import get_session_manager

class TaskExtractorAgent:
    """Agent responsible for extracting tasks and action items from conversations."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.llm = ChatVertexAI(
            model_name=model_config.get("model_name", "gemini-1.5-flash"),
            max_output_tokens=model_config.get("max_tokens", 2000),
            temperature=model_config.get("temperature", 0.1),
            top_p=model_config.get("top_p", 0.95)
        )
        
        # Task indication keywords
        self.task_indicators = [
            "todo", "task", "action item", "follow up", "need to", "should", 
            "will do", "can you", "please", "assign", "deadline", "due",
            "by when", "schedule", "plan", "implement", "create", "build",
            "fix", "review", "update", "check", "test", "deploy"
        ]
    
    def _format_conversation_for_prompt(self, conversation: ConversationData) -> str:
        """Format conversation data focusing on actionable content."""
        
        formatted_text = f"Channel: #{conversation.channel_name} ({conversation.source})\n"
        formatted_text += f"Participants: {conversation.participant_count} people\n\n"
        formatted_text += "Messages:\n"
        
        for message in conversation.messages:
            timestamp = message.timestamp.strftime("%H:%M")
            formatted_text += f"[{timestamp}] {message.author}: {message.content}\n"
        
        return formatted_text
    
    @observe(as_type="generation")
    def extract_tasks(self, conversation: ConversationData) -> List[TaskItem]:
        """Extract tasks and action items from a conversation."""
        
        # Update current trace with session information
        try:
            session_manager = get_session_manager()
            session_manager.update_current_trace_with_session()
        except Exception:
            pass  # Continue if session management fails
        
        conversation_text = self._format_conversation_for_prompt(conversation)
        
        system_prompt = """You are an expert at identifying tasks, action items, and commitments in team conversations. Look for:

1. Explicit tasks ("I'll do X", "Can you handle Y", "We need to Z")
2. Commitments with deadlines ("by Friday", "next week", "before the meeting")
3. Assignments to specific people
4. Follow-up items that were mentioned
5. Decisions that require implementation

For each task you identify, provide:
- TITLE: Brief descriptive title
- DESCRIPTION: Clear description of what needs to be done
- ASSIGNEE: Person responsible (if mentioned)
- PRIORITY: high/medium/low based on urgency and importance
- DUE_DATE: Any mentioned deadlines (use format: YYYY-MM-DD)
- SOURCE_MESSAGES: The author names who mentioned this task
- CONFIDENCE: Your confidence in this being a real task (0.0-1.0)

Format each task as:
TASK_START
TITLE: [title]
DESCRIPTION: [description]
ASSIGNEE: [person or "unassigned"]
PRIORITY: [high/medium/low]
DUE_DATE: [date or "none"]
SOURCE_MESSAGES: [author names]
CONFIDENCE: [0.0-1.0]
TASK_END"""

        human_prompt = f"""Please extract all tasks and action items from this conversation:

{conversation_text}

Be thorough but only include genuine action items that require follow-up. Ignore casual mentions or hypothetical discussions."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.llm.invoke(messages)
        tasks = self._parse_task_response(response.content, conversation)
        
        return tasks
    
    def _parse_task_response(self, response_text: str, conversation: ConversationData) -> List[TaskItem]:
        """Parse the LLM response into TaskItem objects."""
        
        tasks = []
        lines = response_text.split('\n')
        
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
                    current_task["description"] = line.replace("DESCRIPTION:", "").strip()
                elif line.startswith("ASSIGNEE:"):
                    current_task["assignee"] = line.replace("ASSIGNEE:", "").strip()
                elif line.startswith("PRIORITY:"):
                    current_task["priority"] = line.replace("PRIORITY:", "").strip()
                elif line.startswith("DUE_DATE:"):
                    current_task["due_date"] = line.replace("DUE_DATE:", "").strip()
                elif line.startswith("SOURCE_MESSAGES:"):
                    current_task["source_messages"] = line.replace("SOURCE_MESSAGES:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    current_task["confidence"] = line.replace("CONFIDENCE:", "").strip()
        
        return tasks
    
    def _create_task_item(self, task_data: Dict[str, str], conversation: ConversationData) -> TaskItem:
        """Create a TaskItem from parsed data."""
        
        try:
            # Parse due date
            due_date = None
            if task_data.get("due_date") and task_data["due_date"].lower() not in ["none", "unspecified"]:
                try:
                    due_date = datetime.strptime(task_data["due_date"], "%Y-%m-%d")
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
                if any(author.strip().lower() in message.author.lower() for author in source_authors):
                    source_messages.append(message.id)
            
            return TaskItem(
                title=task_data.get("title", "Untitled Task"),
                description=task_data.get("description", ""),
                assignee=assignee,
                priority=task_data.get("priority", "medium").lower(),
                due_date=due_date,
                source_messages=source_messages,
                confidence_score=confidence_score
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
    
    @observe(as_type="generation")
    def extract_tasks_from_multiple_conversations(self, conversations: List[ConversationData]) -> List[TaskItem]:
        """Extract tasks from multiple conversations."""
        
        # Update current trace with session information
        try:
            session_manager = get_session_manager()
            session_manager.update_current_trace_with_session()
        except Exception:
            pass  # Continue if session management fails
        
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
                title_similarity = self._calculate_similarity(task.title.lower(), existing_task.title.lower())
                desc_similarity = self._calculate_similarity(task.description.lower(), existing_task.description.lower())
                
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