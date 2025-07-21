"""
LangGraph agent coordination step for orchestrating multi-agent processing.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from langfuse import observe
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from zenml import step
from zenml.logger import get_logger
from zenml.types import HTMLString

from ..agents.summarizer_agent import SummarizerAgent
from ..agents.task_extractor_agent import TaskExtractorAgent
from ..utils.models import (
    CleanedConversationData,
    ProcessedData,
    Summary,
    TaskItem,
)

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State object passed between agents in the LangGraph workflow."""
    
    conversations: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    processing_metadata: Dict[str, Any]
    llm_usage_stats: Dict[str, Any]
    current_step: str
    messages: Annotated[list, add_messages]


class LangGraphOrchestrator:
    """Orchestrates the multi-agent workflow using LangGraph."""

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the LangGraph orchestrator."""
        self.model_config = model_config
        self.summarizer_agent = SummarizerAgent(model_config)
        self.task_extractor_agent = TaskExtractorAgent(model_config)
        self.trace_id = str(uuid.uuid4())
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes (agent functions)
        workflow.add_node("initialize", self._initialize_processing)
        workflow.add_node("summarize", self._summarize_conversations)
        workflow.add_node("extract_tasks", self._extract_tasks)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("finalize", self._finalize_processing)
        
        # Define the workflow edges
        workflow.add_edge("initialize", "summarize")
        workflow.add_edge("summarize", "extract_tasks")
        workflow.add_edge("extract_tasks", "quality_check")
        workflow.add_edge("quality_check", "finalize")
        workflow.add_edge("finalize", END)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        return workflow.compile()
    
    @observe(as_type="span")
    def _initialize_processing(self, state: AgentState) -> AgentState:
        """Initialize the processing workflow."""
        logger.info("Initializing LangGraph processing workflow")
        
        state["processing_metadata"] = {
            "start_time": datetime.utcnow().isoformat(),
            "trace_id": self.trace_id,
            "total_conversations": len(state["conversations"]),
            "workflow_version": "1.0"
        }
        
        state["llm_usage_stats"] = {
            "total_tokens": 0,
            "summarization_tokens": 0,
            "task_extraction_tokens": 0,
            "api_calls": 0
        }
        
        state["current_step"] = "initialized"
        state["messages"].append({
            "role": "system",
            "content": f"Initialized processing for {len(state['conversations'])} conversations"
        })
        
        return state
    
    @observe(as_type="span")
    def _summarize_conversations(self, state: AgentState) -> AgentState:
        """Summarize all conversations using the summarizer agent."""
        logger.info(f"Starting summarization for {len(state['conversations'])} conversations")
        state["current_step"] = "summarizing"
        
        summaries = []
        
        # Convert conversations back to ConversationData objects
        conversation_objects = []
        for conv_data in state["conversations"]:
            # This would require proper deserialization in a real implementation
            # For now, we'll work with the dict data directly
            conversation_objects.append(conv_data)
        
        try:
            # Create individual summaries
            for i, conversation in enumerate(conversation_objects):
                logger.info(f"Summarizing conversation {i+1}/{len(conversation_objects)}")
                
                # Convert dict back to ConversationData object for agent
                # In a real implementation, you'd have proper serialization/deserialization
                from ..utils.models import ChatMessage, ConversationData
                
                messages = [
                    ChatMessage(**msg_data) for msg_data in conversation["messages"]
                ]
                
                conv_obj = ConversationData(
                    messages=messages,
                    channel_name=conversation["channel_name"],
                    source=conversation["source"],
                    date_range=conversation["date_range"],
                    participant_count=conversation["participant_count"],
                    total_messages=conversation["total_messages"]
                )
                
                summary = self.summarizer_agent.create_summary(conv_obj)
                summaries.append(summary.dict())
                
                # Update usage stats
                state["llm_usage_stats"]["api_calls"] += 1
                state["llm_usage_stats"]["summarization_tokens"] += len(summary.content.split())
            
            # Create a combined daily summary if multiple conversations
            if len(conversation_objects) > 1:
                conv_objs = []
                for conv_data in conversation_objects:
                    messages = [ChatMessage(**msg_data) for msg_data in conv_data["messages"]]
                    conv_obj = ConversationData(
                        messages=messages,
                        channel_name=conv_data["channel_name"],
                        source=conv_data["source"],
                        date_range=conv_data["date_range"],
                        participant_count=conv_data["participant_count"],
                        total_messages=conv_data["total_messages"]
                    )
                    conv_objs.append(conv_obj)
                
                daily_summary = self.summarizer_agent.create_multi_conversation_summary(conv_objs)
                summaries.append(daily_summary.dict())
                
                state["llm_usage_stats"]["api_calls"] += 1
                state["llm_usage_stats"]["summarization_tokens"] += len(daily_summary.content.split())
            
            state["summaries"] = summaries
            
            logger.info(f"Summarization complete: {len(summaries)} summaries generated")
            state["messages"].append({
                "role": "assistant",
                "content": f"Generated {len(summaries)} summaries from conversations"
            })
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            state["messages"].append({
                "role": "system",
                "content": f"Error during summarization: {str(e)}"
            })
        
        return state
    
    @observe(as_type="span")
    def _extract_tasks(self, state: AgentState) -> AgentState:
        """Extract tasks from conversations using the task extractor agent."""
        logger.info("Starting task extraction")
        state["current_step"] = "extracting_tasks"
        
        all_tasks = []
        
        try:
            # Convert conversations back to ConversationData objects
            conversation_objects = []
            for conv_data in state["conversations"]:
                from ..utils.models import ChatMessage, ConversationData
                
                messages = [ChatMessage(**msg_data) for msg_data in conv_data["messages"]]
                conv_obj = ConversationData(
                    messages=messages,
                    channel_name=conv_data["channel_name"],
                    source=conv_data["source"],
                    date_range=conv_data["date_range"],
                    participant_count=conv_data["participant_count"],
                    total_messages=conv_data["total_messages"]
                )
                conversation_objects.append(conv_obj)
            
            # Extract tasks from all conversations
            tasks = self.task_extractor_agent.extract_tasks_from_multiple_conversations(conversation_objects)
            all_tasks = [task.dict() for task in tasks]
            
            state["tasks"] = all_tasks
            state["llm_usage_stats"]["api_calls"] += len(conversation_objects)
            state["llm_usage_stats"]["task_extraction_tokens"] += sum(len(task["description"].split()) for task in all_tasks)
            
            logger.info(f"Task extraction complete: {len(all_tasks)} tasks identified")
            state["messages"].append({
                "role": "assistant", 
                "content": f"Extracted {len(all_tasks)} tasks and action items"
            })
            
        except Exception as e:
            logger.error(f"Error during task extraction: {e}")
            state["tasks"] = []
            state["messages"].append({
                "role": "system",
                "content": f"Error during task extraction: {str(e)}"
            })
        
        return state
    
    @observe(as_type="span")
    def _quality_check(self, state: AgentState) -> AgentState:
        """Perform quality checks on generated summaries and tasks."""
        logger.info("Performing quality checks")
        state["current_step"] = "quality_checking"
        
        quality_issues = []
        
        # Check summary quality
        for summary in state["summaries"]:
            if len(summary["content"]) < 50:
                quality_issues.append(f"Summary '{summary['title']}' is too short")
            if summary["confidence_score"] < 0.5:
                quality_issues.append(f"Low confidence summary: {summary['title']}")
        
        # Check task quality
        for task in state["tasks"]:
            if len(task["description"]) < 10:
                quality_issues.append(f"Task '{task['title']}' has insufficient description")
            if task["confidence_score"] < 0.3:
                quality_issues.append(f"Low confidence task: {task['title']}")
        
        # Log quality issues
        if quality_issues:
            logger.warning(f"Quality issues found: {quality_issues}")
            state["processing_metadata"]["quality_issues"] = quality_issues
        else:
            logger.info("Quality check passed")
            state["processing_metadata"]["quality_issues"] = []
        
        state["messages"].append({
            "role": "system",
            "content": f"Quality check complete. Found {len(quality_issues)} issues."
        })
        
        return state
    
    @observe(as_type="span")
    def _finalize_processing(self, state: AgentState) -> AgentState:
        """Finalize the processing workflow."""
        logger.info("Finalizing processing")
        state["current_step"] = "completed"
        
        # Calculate final usage statistics
        state["llm_usage_stats"]["total_tokens"] = (
            state["llm_usage_stats"]["summarization_tokens"] + 
            state["llm_usage_stats"]["task_extraction_tokens"]
        )
        
        state["processing_metadata"]["end_time"] = datetime.utcnow().isoformat()
        
        # Calculate processing time
        start_time = datetime.fromisoformat(state["processing_metadata"]["start_time"])
        end_time = datetime.fromisoformat(state["processing_metadata"]["end_time"])
        processing_time = (end_time - start_time).total_seconds()
        state["processing_metadata"]["processing_time_seconds"] = processing_time
        
        logger.info(f"Processing complete in {processing_time:.2f} seconds")
        state["messages"].append({
            "role": "system",
            "content": f"Processing completed successfully in {processing_time:.2f} seconds"
        })
        
        return state


def _create_agent_processing_visualization(processed_data: ProcessedData, cleaned_data: CleanedConversationData) -> HTMLString:
    """Create HTML visualization for LangGraph agent processing results."""
    total_messages = sum(len(conv.messages) for conv in cleaned_data.conversations)
    avg_confidence_summaries = sum(s.confidence_score for s in processed_data.summaries) / len(processed_data.summaries) if processed_data.summaries else 0
    avg_confidence_tasks = sum(t.confidence_score for t in processed_data.tasks) / len(processed_data.tasks) if processed_data.tasks else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Agent Processing Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
            .metric-value {{ font-weight: bold; color: #2196F3; }}
            .summary-item {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; border-radius: 4px; }}
            .task-item {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; border-radius: 4px; }}
            .high-priority {{ border-left-color: #dc3545; }}
            .medium-priority {{ border-left-color: #ffc107; }}
            .low-priority {{ border-left-color: #28a745; }}
            h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
            .agent-trace {{ background: #e3f2fd; padding: 10px; border-radius: 4px; font-family: monospace; }}
            .confidence {{ float: right; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 LangGraph Agent Processing Results</h1>
            <p><strong>Trace ID:</strong> <span class="agent-trace">{processed_data.agent_trace_id}</span></p>
            <p><strong>Processed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary-grid">
                <div class="card">
                    <h2>📊 Processing Overview</h2>
                    <div class="metric">
                        <span>Input Messages:</span>
                        <span class="metric-value">{total_messages}</span>
                    </div>
                    <div class="metric">
                        <span>Conversations:</span>
                        <span class="metric-value">{len(cleaned_data.conversations)}</span>
                    </div>
                    <div class="metric">
                        <span>Summaries Generated:</span>
                        <span class="metric-value">{len(processed_data.summaries)}</span>
                    </div>
                    <div class="metric">
                        <span>Tasks Extracted:</span>
                        <span class="metric-value">{len(processed_data.tasks)}</span>
                    </div>
                    <div class="metric">
                        <span>Processing Time:</span>
                        <span class="metric-value">{processed_data.processing_metadata.get('processing_time_seconds', 0):.1f}s</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🎯 Agent Performance</h2>
                    <div class="metric">
                        <span>Summary Confidence:</span>
                        <span class="metric-value">{avg_confidence_summaries:.1%}</span>
                    </div>
                    <div class="metric">
                        <span>Task Confidence:</span>
                        <span class="metric-value">{avg_confidence_tasks:.1%}</span>
                    </div>
                    <div class="metric">
                        <span>API Calls:</span>
                        <span class="metric-value">{processed_data.llm_usage_stats.get('api_calls', 0)}</span>
                    </div>
                    <div class="metric">
                        <span>Total Tokens:</span>
                        <span class="metric-value">{processed_data.llm_usage_stats.get('total_tokens', 0):,}</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📝 Generated Summaries</h2>
                {''.join([f'''
                <div class="summary-item">
                    <h3>{cleaned_data.conversations[i].channel_name if i < len(cleaned_data.conversations) else "Unknown Channel"} Summary <span class="confidence">{summary.confidence_score:.1%} confidence</span></h3>
                    <p><strong>Topics:</strong> {', '.join(summary.topics[:3])}{'...' if len(summary.topics) > 3 else ''}</p>
                    <p><strong>Key Points:</strong> {len(summary.key_points)} identified</p>
                    <p><strong>Participants:</strong> {', '.join(summary.participants[:5])}{'...' if len(summary.participants) > 5 else ''}</p>
                    <p><strong>Word Count:</strong> {summary.word_count} words</p>
                </div>
                ''' for i, summary in enumerate(processed_data.summaries[:3])])}
                {f'<p><em>... and {len(processed_data.summaries) - 3} more summaries</em></p>' if len(processed_data.summaries) > 3 else ''}
            </div>
            
            <div class="card">
                <h2>✅ Extracted Tasks</h2>
                {''.join([f'''
                <div class="task-item {task.priority}-priority">
                    <h3>{task.title} <span class="confidence">{task.confidence_score:.1%} confidence</span></h3>
                    <p><strong>Priority:</strong> {task.priority.upper()}</p>
                    <p><strong>Assigned to:</strong> {task.assignee or 'Unassigned'}</p>
                    <p><strong>Due:</strong> {task.due_date or 'No due date'}</p>
                    <p><strong>Description:</strong> {task.description[:100]}{'...' if len(task.description) > 100 else ''}</p>
                </div>
                ''' for task in processed_data.tasks[:5]])}
                {f'<p><em>... and {len(processed_data.tasks) - 5} more tasks</em></p>' if len(processed_data.tasks) > 5 else ''}
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLString(html_content)


@step
def langgraph_agent_step(
    cleaned_data: CleanedConversationData,
    model_config: Dict[str, Any]
) -> Tuple[ProcessedData, HTMLString]:
    """Process cleaned conversation data using LangGraph multi-agent workflow.
    
    Args:
        cleaned_data: Cleaned conversation data from preprocessing
        model_config: Configuration for the LLM models
        
    Returns:
        ProcessedData: Processed summaries and tasks from agent workflow
    """
    logger.info(f"Starting LangGraph agent processing for {len(cleaned_data.conversations)} conversations")
    
    # Initialize the orchestrator
    orchestrator = LangGraphOrchestrator(model_config)
    
    # Prepare initial state
    initial_state = AgentState(
        conversations=[conv.dict() for conv in cleaned_data.conversations],
        summaries=[],
        tasks=[],
        processing_metadata={},
        llm_usage_stats={},
        current_step="",
        messages=[]
    )
    
    # Run the workflow
    final_state = orchestrator.workflow.invoke(initial_state)
    
    # Convert results back to our data models
    summaries = [Summary(**summary_data) for summary_data in final_state["summaries"]]
    tasks = [TaskItem(**task_data) for task_data in final_state["tasks"]]
    
    processed_data = ProcessedData(
        summaries=summaries,
        tasks=tasks,
        processing_metadata=final_state["processing_metadata"],
        llm_usage_stats=final_state["llm_usage_stats"],
        agent_trace_id=orchestrator.trace_id
    )
    
    logger.info(f"LangGraph processing complete: {len(summaries)} summaries, {len(tasks)} tasks")
    
    # Generate HTML visualization
    html_viz = _create_agent_processing_visualization(processed_data, cleaned_data)
    
    return processed_data, html_viz