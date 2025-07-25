"""
LangGraph agent coordination step for orchestrating multi-agent processing.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from zenml import step
from zenml.logger import get_logger
from zenml.types import HTMLString

from ..agents.summarizer_agent import SummarizerAgent
from ..agents.task_extractor_agent import TaskExtractorAgent
from ..utils.models import (
    ProcessedData,
    RawConversationData,
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
    errors: List[str]  # NEW: collect non-fatal errors
    messages: Annotated[list, add_messages]


class LangGraphOrchestrator:
    """Orchestrates the multi-agent workflow using LangGraph."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        *,
        extract_tasks: bool = True,
        max_workers: int = 4,
    ):
        """Initialize the LangGraph orchestrator."""
        self.model_config = model_config
        self.extract_tasks = extract_tasks
        self.max_workers = max_workers

        # Pass concurrency limits down to the agents
        self.summarizer_agent = SummarizerAgent(
            model_config, max_workers=max_workers
        )
        self.task_extractor_agent = (
            TaskExtractorAgent(model_config, max_workers=max_workers)
            if self.extract_tasks
            else None
        )
        self.trace_id = str(uuid.uuid4())

        # Get ZenML pipeline run ID for tagging
        try:
            from zenml import get_step_context

            step_context = get_step_context()
            self.run_id = str(step_context.pipeline_run.id)
        except Exception:
            self.run_id = str(uuid.uuid4())

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("initialize", self._initialize_processing)
        workflow.add_node("summarize", self._summarize_conversations)

        if self.extract_tasks:
            workflow.add_node("extract_tasks", self._extract_tasks)

        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("finalize", self._finalize_processing)

        # Define edges
        workflow.add_edge("initialize", "summarize")

        if self.extract_tasks:
            workflow.add_edge("summarize", "extract_tasks")
            workflow.add_edge("extract_tasks", "quality_check")
        else:
            workflow.add_edge("summarize", "quality_check")

        workflow.add_edge("quality_check", "finalize")
        workflow.add_edge("finalize", END)

        # Entry point
        workflow.set_entry_point("initialize")

        return workflow.compile()

    def _initialize_processing(self, state: AgentState) -> AgentState:
        """Initialize the processing workflow."""
        logger.info(
            f"Initializing LangGraph processing workflow with ZenML run ID: {self.run_id}"
        )

        state["processing_metadata"] = {
            "start_time": datetime.utcnow().isoformat(),
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "zenml_run_name": "unknown",
            "zenml_pipeline_name": "unknown",
            "total_conversations": len(state["conversations"]),
            "workflow_version": "1.0",
        }

        state["llm_usage_stats"] = {
            "total_tokens": 0,
            "summarization_tokens": 0,
            "task_extraction_tokens": 0,
            "api_calls": 0,
        }

        state["current_step"] = "initialized"
        state["messages"].append(
            {
                "role": "system",
                "content": f"Initialized processing for {len(state['conversations'])} conversations",
            }
        )

        return state

    def _summarize_conversations(self, state: AgentState) -> AgentState:
        """Summarize all conversations using the summarizer agent."""
        logger.info(
            f"Starting summarization for {len(state['conversations'])} conversations"
        )
        state["current_step"] = "summarizing"

        summaries = []
        summarize_errors: List[str] = []  # NEW

        # Convert conversations back to ConversationData objects
        conversation_objects = []
        for conv_data in state["conversations"]:
            # This would require proper deserialization in a real implementation
            # For now, we'll work with the dict data directly
            conversation_objects.append(conv_data)

        try:
            # Create individual summaries
            for i, conversation in enumerate(conversation_objects):
                logger.info(
                    f"Summarizing conversation {i+1}/{len(conversation_objects)}"
                )

                # Convert dict back to ConversationData object for agent
                # In a real implementation, you'd have proper serialization/deserialization
                from ..utils.models import ChatMessage, ConversationData

                messages = [
                    ChatMessage(**msg_data)
                    for msg_data in conversation["messages"]
                ]

                conv_obj = ConversationData(
                    messages=messages,
                    channel_name=conversation["channel_name"],
                    source=conversation["source"],
                    date_range=conversation["date_range"],
                    participant_count=conversation["participant_count"],
                    total_messages=conversation["total_messages"],
                )

                summary = self.summarizer_agent.create_summary(conv_obj)
                summaries.append(summary.dict())

                # Update usage stats
                state["llm_usage_stats"]["api_calls"] += 1
                state["llm_usage_stats"]["summarization_tokens"] += len(
                    summary.content.split()
                )

            # Create a combined daily summary if multiple conversations
            if len(conversation_objects) > 1:
                conv_objs = []
                for conv_data in conversation_objects:
                    messages = [
                        ChatMessage(**msg_data)
                        for msg_data in conv_data["messages"]
                    ]
                    conv_obj = ConversationData(
                        messages=messages,
                        channel_name=conv_data["channel_name"],
                        source=conv_data["source"],
                        date_range=conv_data["date_range"],
                        participant_count=conv_data["participant_count"],
                        total_messages=conv_data["total_messages"],
                    )
                    conv_objs.append(conv_obj)

                combined_summary, errors = (
                    self.summarizer_agent.create_multi_conversation_summary(
                        conv_objs
                    )
                )
                summaries.append(combined_summary.dict())
                summarize_errors.extend(errors)  # NEW collect any errors

                state["llm_usage_stats"]["api_calls"] += 1
                state["llm_usage_stats"]["summarization_tokens"] += len(
                    combined_summary.content.split()
                )

            state["summaries"] = summaries
            # Merge newly gathered errors into state
            state.setdefault("errors", []).extend(summarize_errors)  # NEW

            logger.info(
                f"Summarization complete: {len(summaries)} summaries generated"
            )
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Generated {len(summaries)} summaries from conversations",
                }
            )

        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            state["messages"].append(
                {
                    "role": "system",
                    "content": f"Error during summarization: {str(e)}",
                }
            )

        return state

    def _extract_tasks(self, state: AgentState) -> AgentState:
        """Extract tasks from conversations using the task extractor agent."""
        if not self.extract_tasks:
            # Skip extraction; ensure tasks key exists
            state["tasks"] = []
            return state

        logger.info("Starting task extraction")
        state["current_step"] = "extracting_tasks"

        all_tasks = []
        extraction_errors: List[str] = []  # NEW

        try:
            # Convert conversations back to ConversationData objects
            conversation_objects = []
            for conv_data in state["conversations"]:
                from ..utils.models import ChatMessage, ConversationData

                messages = [
                    ChatMessage(**msg_data)
                    for msg_data in conv_data["messages"]
                ]
                conv_obj = ConversationData(
                    messages=messages,
                    channel_name=conv_data["channel_name"],
                    source=conv_data["source"],
                    date_range=conv_data["date_range"],
                    participant_count=conv_data["participant_count"],
                    total_messages=conv_data["total_messages"],
                )
                conversation_objects.append(conv_obj)

            # Extract tasks from all conversations
            tasks, errors = (
                self.task_extractor_agent.extract_tasks_from_multiple_conversations(  # type: ignore
                    conversation_objects
                )
            )
            all_tasks = [task.dict() for task in tasks]
            extraction_errors.extend(errors)  # NEW

            state["tasks"] = all_tasks
            state["llm_usage_stats"]["api_calls"] += len(conversation_objects)
            state["llm_usage_stats"]["task_extraction_tokens"] += sum(
                len(task["description"].split()) for task in all_tasks
            )

            # Append newly collected errors
            state.setdefault("errors", []).extend(extraction_errors)  # NEW

            logger.info(
                f"Task extraction complete: {len(all_tasks)} tasks identified"
            )
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Extracted {len(all_tasks)} tasks and action items",
                }
            )

        except Exception as e:
            logger.error(f"Error during task extraction: {e}")
            state["tasks"] = []
            state["messages"].append(
                {
                    "role": "system",
                    "content": f"Error during task extraction: {str(e)}",
                }
            )

        return state

    def _quality_check(self, state: AgentState) -> AgentState:
        """Perform quality checks on generated summaries and tasks."""
        logger.info("Performing quality checks")
        state["current_step"] = "quality_checking"

        quality_issues = []

        # Check summary quality
        for summary in state["summaries"]:
            if len(summary["content"]) < 50:
                quality_issues.append(
                    f"Summary '{summary['title']}' is too short"
                )
            if summary["confidence_score"] < 0.5:
                quality_issues.append(
                    f"Low confidence summary: {summary['title']}"
                )

        # Check task quality
        for task in state["tasks"]:
            if len(task["description"]) < 10:
                quality_issues.append(
                    f"Task '{task['title']}' has insufficient description"
                )
            if task["confidence_score"] < 0.3:
                quality_issues.append(f"Low confidence task: {task['title']}")

        # Log quality issues
        if quality_issues:
            logger.warning(f"Quality issues found: {quality_issues}")
            state["processing_metadata"]["quality_issues"] = quality_issues
        else:
            logger.info("Quality check passed")
            state["processing_metadata"]["quality_issues"] = []

        state["messages"].append(
            {
                "role": "system",
                "content": f"Quality check complete. Found {len(quality_issues)} issues.",
            }
        )

        return state

    def _finalize_processing(self, state: AgentState) -> AgentState:
        """Finalize the processing workflow."""
        logger.info("Finalizing processing")
        state["current_step"] = "completed"

        # Calculate final usage statistics
        state["llm_usage_stats"]["total_tokens"] = (
            state["llm_usage_stats"]["summarization_tokens"]
            + state["llm_usage_stats"]["task_extraction_tokens"]
        )

        state["processing_metadata"]["end_time"] = (
            datetime.utcnow().isoformat()
        )

        # Calculate processing time
        start_time = datetime.fromisoformat(
            state["processing_metadata"]["start_time"]
        )
        end_time = datetime.fromisoformat(
            state["processing_metadata"]["end_time"]
        )
        processing_time = (end_time - start_time).total_seconds()
        state["processing_metadata"]["processing_time_seconds"] = (
            processing_time
        )

        logger.info(f"Processing complete in {processing_time:.2f} seconds")
        state["messages"].append(
            {
                "role": "system",
                "content": f"Processing completed successfully in {processing_time:.2f} seconds",
            }
        )

        return state


def _create_agent_architecture_visualization(
    processed_data: ProcessedData, orchestrator: "LangGraphOrchestrator"
) -> HTMLString:
    """Create HTML visualization for LangGraph agent architecture."""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Agent Architecture</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            
            .workflow-diagram {{
                background: #f8f9fa;
                padding: 30px;
                margin: 30px;
                border-radius: 10px;
            }}
            
            .workflow-steps {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 20px;
            }}
            
            .workflow-step {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                text-align: center;
                flex: 1;
                min-width: 150px;
                border-left: 4px solid;
            }}
            
            .step-initialize {{ border-left-color: #3498db; }}
            .step-summarize {{ border-left-color: #e74c3c; }}
            .step-extract {{ border-left-color: #f39c12; }}
            .step-quality {{ border-left-color: #9b59b6; }}
            .step-finalize {{ border-left-color: #27ae60; }}
            
            .workflow-arrow {{
                font-size: 2em;
                color: #bdc3c7;
            }}
            
            .agents-section {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin: 30px;
            }}
            
            .agent-card {{
                background: white;
                border-radius: 10px;
                padding: 25px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border-top: 5px solid;
            }}
            
            .summarizer-agent {{ border-top-color: #e74c3c; }}
            .task-agent {{ border-top-color: #f39c12; }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px;
            }}
            
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            
            .metric-value {{
                font-size: 2.5em;
                font-weight: 700;
                color: #3498db;
                margin: 10px 0;
            }}
            
            .metric-label {{
                color: #666;
                font-size: 1em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 LangGraph Agent Architecture</h1>
                <p>Multi-Agent Workflow Orchestration</p>
                <p><strong>Trace ID:</strong> {processed_data.agent_trace_id}</p>
            </div>
            
            <div class="workflow-diagram">
                <h2 style="text-align: center; color: #2c3e50; margin-bottom: 30px;">🔄 Workflow Pipeline</h2>
                <div class="workflow-steps">
                    <div class="workflow-step step-initialize">
                        <h3>🚀 Initialize</h3>
                        <p>Setup metadata and state</p>
                    </div>
                    <div class="workflow-arrow">→</div>
                    
                    <div class="workflow-step step-summarize">
                        <h3>📝 Summarize</h3>
                        <p>Generate summaries via SummarizerAgent</p>
                    </div>
                    <div class="workflow-arrow">→</div>
                    
                    <div class="workflow-step step-extract">
                        <h3>✅ Extract Tasks</h3>
                        <p>Identify tasks via TaskExtractorAgent</p>
                    </div>
                    <div class="workflow-arrow">→</div>
                    
                    <div class="workflow-step step-quality">
                        <h3>🔍 Quality Check</h3>
                        <p>Validate output quality</p>
                    </div>
                    <div class="workflow-arrow">→</div>
                    
                    <div class="workflow-step step-finalize">
                        <h3>🏁 Finalize</h3>
                        <p>Complete processing</p>
                    </div>
                </div>
            </div>
            
            <div class="agents-section">
                <div class="agent-card summarizer-agent">
                    <h3>📝 Summarizer Agent</h3>
                    <p><strong>Model:</strong> {orchestrator.model_config.get('model_name', 'gemini-2.5-flash')}</p>
                    <p><strong>Temperature:</strong> {orchestrator.model_config.get('temperature', 0.1)}</p>
                    <p><strong>Max Tokens:</strong> {orchestrator.model_config.get('max_tokens', 4000)}</p>
                    <p><strong>Features:</strong> Multi-conversation processing, structured output parsing, confidence scoring</p>
                </div>
                
                <div class="agent-card task-agent">
                    <h3>✅ Task Extractor Agent</h3>
                    <p><strong>Keywords:</strong> 32 task indicators</p>
                    <p><strong>Parsing:</strong> TASK_START/END format</p>
                    <p><strong>Deduplication:</strong> 0.8 similarity threshold</p>
                    <p><strong>Features:</strong> Assignment detection, deadline parsing, priority classification</p>
                </div>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(processed_data.summaries)}</div>
                    <div class="metric-label">Summaries Generated</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(processed_data.tasks)}</div>
                    <div class="metric-label">Tasks Extracted</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{processed_data.llm_usage_stats.get('api_calls', 0)}</div>
                    <div class="metric-label">API Calls</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{processed_data.llm_usage_stats.get('total_tokens', 0):,}</div>
                    <div class="metric-label">Total Tokens</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html_content)


def _create_langgraph_traces_visualization(
    processed_data: ProcessedData, final_state: AgentState
) -> HTMLString:
    """Create HTML visualization for LangGraph execution traces."""

    # Extract timing and execution data from the final state
    processing_time = processed_data.processing_metadata.get(
        "processing_time_seconds", 0
    )
    start_time = processed_data.processing_metadata.get(
        "start_time", datetime.now().isoformat()
    )
    end_time = processed_data.processing_metadata.get(
        "end_time", datetime.now().isoformat()
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Execution Traces</title>
        <style>
            body {{
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                margin: 0;
                padding: 20px;
                background: #0d1117;
                color: #c9d1d9;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background: #161b22;
                border-radius: 12px;
                box-shadow: 0 16px 32px rgba(0,0,0,0.4);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #1f6feb 0%, #8b5cf6 100%);
                padding: 30px;
                text-align: center;
                color: white;
            }}
            
            .trace-metadata {{
                background: #21262d;
                padding: 20px;
                margin: 20px;
                border-radius: 8px;
                border-left: 4px solid #f85149;
            }}
            
            .metadata-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            
            .metadata-item {{
                background: #0d1117;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #30363d;
            }}
            
            .metadata-label {{
                color: #7d8590;
                font-size: 0.9em;
            }}
            
            .metadata-value {{
                color: #58a6ff;
                font-weight: 600;
            }}
            
            .trace-timeline {{
                position: relative;
                margin: 30px 20px;
            }}
            
            .timeline-line {{
                position: absolute;
                left: 30px;
                top: 0;
                bottom: 0;
                width: 2px;
                background: linear-gradient(180deg, #1f6feb 0%, #8b5cf6 100%);
            }}
            
            .trace-step {{
                position: relative;
                margin: 0 0 25px 70px;
                background: #21262d;
                border-radius: 10px;
                border: 1px solid #30363d;
            }}
            
            .step-icon {{
                position: absolute;
                left: -55px;
                top: 15px;
                width: 30px;
                height: 30px;
                background: #1f6feb;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                border: 3px solid #0d1117;
            }}
            
            .step-header {{
                padding: 20px;
                border-bottom: 1px solid #30363d;
            }}
            
            .step-title {{
                color: #f0f6fc;
                font-size: 1.3em;
                margin: 0;
            }}
            
            .step-content {{
                padding: 20px;
            }}
            
            .langfuse-trace {{
                background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }}
            
            .json-viewer {{
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 15px;
                font-family: 'Monaco', monospace;
                font-size: 0.9em;
                overflow-x: auto;
            }}
            
            .json-key {{ color: #79c0ff; }}
            .json-string {{ color: #a5d6ff; }}
            .json-number {{ color: #79c0ff; }}
            .json-boolean {{ color: #ff7b72; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 LangGraph Execution Traces</h1>
                <p>Real-time workflow execution monitoring</p>
            </div>
            
            <div class="trace-metadata">
                <h2 style="color: #f85149; margin: 0 0 15px 0;">📊 Trace Overview</h2>
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <div class="metadata-label">Trace ID</div>
                        <div class="metadata-value">{processed_data.agent_trace_id}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Total Duration</div>
                        <div class="metadata-value">{processing_time:.2f}s</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">API Calls</div>
                        <div class="metadata-value">{processed_data.llm_usage_stats.get('api_calls', 0)}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Total Tokens</div>
                        <div class="metadata-value">{processed_data.llm_usage_stats.get('total_tokens', 0):,}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Status</div>
                        <div class="metadata-value">✅ Completed</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Quality Issues</div>
                        <div class="metadata-value">{len(processed_data.processing_metadata.get('quality_issues', []))}</div>
                    </div>
                </div>
            </div>
            
            <div class="trace-timeline">
                <div class="timeline-line"></div>
                
                <div class="trace-step">
                    <div class="step-header">
                        <div class="step-icon">🚀</div>
                        <div class="step-title">_initialize_processing</div>
                    </div>
                    <div class="step-content">
                        <div class="langfuse-trace">
                            <strong>@observe(as_type="span")</strong><br>
                            Initialized workflow state and metadata
                        </div>
                        <div class="json-viewer">
<span class="json-key">"processing_metadata"</span>: {{
    <span class="json-key">"start_time"</span>: <span class="json-string">"{start_time}"</span>,
    <span class="json-key">"trace_id"</span>: <span class="json-string">"{processed_data.agent_trace_id}"</span>
}}
                        </div>
                    </div>
                </div>
                
                <div class="trace-step">
                    <div class="step-header">
                        <div class="step-icon">📝</div>
                        <div class="step-title">_summarize_conversations</div>
                    </div>
                    <div class="step-content">
                        <div class="langfuse-trace">
                            <strong>@observe(as_type="generation")</strong><br>
                            SummarizerAgent: Generated {len(processed_data.summaries)} summaries<br>
                            Tokens used: {processed_data.llm_usage_stats.get('summarization_tokens', 0):,}
                        </div>
                    </div>
                </div>
                
                <div class="trace-step">
                    <div class="step-header">
                        <div class="step-icon">✅</div>
                        <div class="step-title">_extract_tasks</div>
                    </div>
                    <div class="step-content">
                        <div class="langfuse-trace">
                            <strong>@observe(as_type="generation")</strong><br>
                            TaskExtractorAgent: Extracted {len(processed_data.tasks)} tasks<br>
                            Tokens used: {processed_data.llm_usage_stats.get('task_extraction_tokens', 0):,}
                        </div>
                    </div>
                </div>
                
                <div class="trace-step">
                    <div class="step-header">
                        <div class="step-icon">🔍</div>
                        <div class="step-title">_quality_check</div>
                    </div>
                    <div class="step-content">
                        <div class="langfuse-trace">
                            <strong>@observe(as_type="span")</strong><br>
                            Quality validation completed<br>
                            Issues found: {len(processed_data.processing_metadata.get('quality_issues', []))}
                        </div>
                    </div>
                </div>
                
                <div class="trace-step">
                    <div class="step-header">
                        <div class="step-icon">🏁</div>
                        <div class="step-title">_finalize_processing</div>
                    </div>
                    <div class="step-content">
                        <div class="langfuse-trace">
                            <strong>@observe(as_type="span")</strong><br>
                            Processing completed successfully<br>
                            End time: {end_time}
                        </div>
                        <div class="json-viewer">
<span class="json-key">"llm_usage_stats"</span>: {{
    <span class="json-key">"total_tokens"</span>: <span class="json-number">{processed_data.llm_usage_stats.get('total_tokens', 0)}</span>,
    <span class="json-key">"api_calls"</span>: <span class="json-number">{processed_data.llm_usage_stats.get('api_calls', 0)}</span>
}}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html_content)


def _create_agent_processing_visualization(
    processed_data: ProcessedData, raw_data: RawConversationData
) -> HTMLString:
    """Create HTML visualization for LangGraph agent processing results."""
    total_messages = sum(len(conv.messages) for conv in raw_data.conversations)
    avg_confidence_summaries = (
        sum(s.confidence_score for s in processed_data.summaries)
        / len(processed_data.summaries)
        if processed_data.summaries
        else 0
    )
    avg_confidence_tasks = (
        sum(t.confidence_score for t in processed_data.tasks)
        / len(processed_data.tasks)
        if processed_data.tasks
        else 0
    )

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
                        <span class="metric-value">{len(raw_data.conversations)}</span>
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
                    <h3>{raw_data.conversations[i].channel_name if i < len(raw_data.conversations) else "Unknown Channel"} Summary <span class="confidence">{summary.confidence_score:.1%} confidence</span></h3>
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


def _create_combined_agent_dashboard(
    architecture_viz: HTMLString,
    traces_viz: HTMLString,
    processing_viz: HTMLString,
) -> HTMLString:
    """Create a combined dashboard with tabs for all agent visualizations."""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Agent Dashboard - LLM Daily Summarization</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            
            .dashboard-container {{
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .dashboard-header {{
                background: white;
                border-radius: 15px 15px 0 0;
                padding: 30px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            
            .dashboard-header h1 {{
                margin: 0 0 10px 0;
                color: #2c3e50;
                font-size: 2.5em;
                font-weight: 300;
            }}
            
            .tab-container {{
                background: white;
                border-radius: 0 0 15px 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .tab-buttons {{
                display: flex;
                background: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }}
            
            .tab-button {{
                flex: 1;
                padding: 15px 25px;
                background: none;
                border: none;
                cursor: pointer;
                font-size: 1.1em;
                color: #6c757d;
                transition: all 0.3s ease;
                border-right: 1px solid #dee2e6;
            }}
            
            .tab-button:last-child {{
                border-right: none;
            }}
            
            .tab-button.active {{
                background: white;
                color: #2c3e50;
                font-weight: 600;
                border-bottom: 3px solid #3498db;
            }}
            
            .tab-button:hover {{
                background: #e9ecef;
                color: #2c3e50;
            }}
            
            .tab-content {{
                display: none;
                min-height: 600px;
            }}
            
            .tab-content.active {{
                display: block;
            }}
            
            iframe {{
                width: 100%;
                height: 800px;
                border: none;
            }}
        </style>
        <script>
            function showTab(tabName, element) {{
                // Hide all tab contents
                var tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(function(content) {{
                    content.classList.remove('active');
                }});
                
                // Remove active class from all buttons
                var tabButtons = document.querySelectorAll('.tab-button');
                tabButtons.forEach(function(button) {{
                    button.classList.remove('active');
                }});
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                element.classList.add('active');
            }}
            
            // Show architecture tab by default
            window.onload = function() {{
                document.querySelector('.tab-button').click();
            }};
        </script>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="dashboard-header">
                <h1>🤖 LangGraph Agent Dashboard</h1>
                <p>Comprehensive visualization of multi-agent workflow execution, architecture, and traces</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button" onclick="showTab('architecture', this)">
                        🏗️ Agent Architecture
                    </button>
                    <button class="tab-button" onclick="showTab('traces', this)">
                        🔍 Execution Traces
                    </button>
                    <button class="tab-button" onclick="showTab('processing', this)">
                        📊 Processing Results
                    </button>
                </div>
                
                <div id="architecture" class="tab-content">
                    <iframe srcdoc="{str(architecture_viz).replace('"', '&quot;')}"></iframe>
                </div>
                
                <div id="traces" class="tab-content">
                    <iframe srcdoc="{str(traces_viz).replace('"', '&quot;')}"></iframe>
                </div>
                
                <div id="processing" class="tab-content">
                    <iframe srcdoc="{str(processing_viz).replace('"', '&quot;')}"></iframe>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html_content)


@step
def langgraph_agent_step(
    raw_data: RawConversationData,
    model_config: Dict[str, Any],
    extract_tasks: bool = True,
) -> Tuple[
    Annotated[ProcessedData, "processed_data"],
    Annotated[HTMLString, "processing_visualization"],
]:
    """Process raw conversation data using LangGraph multi-agent workflow.

    Args:
        raw_data: Raw conversation data from ingestion
        model_config: Configuration for the LLM models

    Returns:
        ProcessedData: Processed summaries and tasks from agent workflow
    """
    logger.info(
        f"Starting LangGraph agent processing for {len(raw_data.conversations)} conversations"
    )

    # Initialize the orchestrator with max_workers from config
    max_workers = model_config.get("max_workers", 4)
    orchestrator = LangGraphOrchestrator(
        model_config,
        extract_tasks=extract_tasks,
        max_workers=max_workers,
    )

    # Prepare initial state
    initial_state = AgentState(
        conversations=[conv.dict() for conv in raw_data.conversations],
        summaries=[],
        tasks=[],
        processing_metadata={},
        llm_usage_stats={},
        current_step="",
        errors=[],  # NEW
        messages=[],
    )

    # Run the workflow
    final_state = orchestrator.workflow.invoke(initial_state)

    # Convert results back to our data models
    summaries = [
        Summary(**summary_data) for summary_data in final_state["summaries"]
    ]
    tasks = [TaskItem(**task_data) for task_data in final_state["tasks"]]

    processed_data = ProcessedData(
        summaries=summaries,
        tasks=tasks,
        processing_metadata=final_state["processing_metadata"],
        llm_usage_stats=final_state["llm_usage_stats"],
        agent_trace_id=orchestrator.trace_id,
        run_id=orchestrator.run_id,
        errors=final_state.get("errors", []),  # NEW: propagate errors
    )

    logger.info(
        f"LangGraph processing complete: {len(summaries)} summaries, {len(tasks)} tasks"
    )

    # Log Langfuse session metadata to ZenML pipeline

    # Generate comprehensive HTML visualizations
    architecture_viz = _create_agent_architecture_visualization(
        processed_data, orchestrator
    )
    traces_viz = _create_langgraph_traces_visualization(
        processed_data, final_state
    )
    processing_viz = _create_agent_processing_visualization(
        processed_data, raw_data
    )

    # Combine all visualizations into a comprehensive dashboard
    combined_viz = _create_combined_agent_dashboard(
        architecture_viz, traces_viz, processing_viz
    )

    return processed_data, combined_viz
