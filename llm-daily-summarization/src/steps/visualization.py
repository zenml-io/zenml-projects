"""
HTML Visualization Step for LLM Daily Summarization Pipeline

This step generates comprehensive HTML visualizations of pipeline artifacts,
including prompts, outputs, conversations, and evaluation metrics.
"""

from typing import Any, Dict, List, Optional

from zenml import step
from zenml.logger import get_logger
from zenml.types import HTMLString

from src.utils.html_visualization import HTMLVisualizer
from src.utils.models import (
    CleanedConversationData,
    ConversationData,
    ProcessedData,
)

logger = get_logger(__name__)


@step
def generate_html_visualization(
    processed_data: ProcessedData,
    raw_conversations: List[ConversationData],
    cleaned_data: List[CleanedConversationData],
    evaluation_metrics: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    agent_prompts: Optional[Dict[str, str]] = None,
) -> HTMLString:
    """
    Generate comprehensive HTML visualization of pipeline results.

    This step creates a professional dashboard showing:
    - Pipeline statistics and overview
    - Generated summaries with confidence scores
    - Extracted tasks with priorities and assignments
    - Raw and cleaned conversation data
    - Agent prompts used for processing
    - Evaluation metrics and performance data

    Args:
        processed_data: Output from LangGraph agent processing
        raw_conversations: Original conversation data from ingestion
        cleaned_data: Preprocessed conversation data
        evaluation_metrics: Optional pipeline evaluation results
        run_metadata: Optional metadata about the pipeline run
        agent_prompts: Optional prompts used by LangGraph agents

    Returns:
        HTMLString: Complete HTML dashboard for visualization
    """
    try:
        logger.info("Generating HTML visualization dashboard...")

        # Initialize the HTML visualizer
        visualizer = HTMLVisualizer()

        # Generate comprehensive dashboard
        html_content = visualizer.generate_pipeline_dashboard(
            processed_data=processed_data,
            raw_conversations=raw_conversations,
            cleaned_data=cleaned_data,
            agent_prompts=agent_prompts,
            evaluation_metrics=evaluation_metrics,
            run_metadata=run_metadata,
        )

        # Log statistics for monitoring
        stats = {
            "conversations": len(raw_conversations),
            "summaries": len(processed_data.summaries),
            "tasks": len(processed_data.tasks),
            "total_tokens": processed_data.usage_stats.get("total_tokens", 0),
            "estimated_cost": processed_data.usage_stats.get("total_cost", 0),
        }

        logger.info(f"HTML visualization generated successfully: {stats}")

        return HTMLString(html_content)

    except Exception as e:
        logger.error(f"Failed to generate HTML visualization: {e}")

        # Return a simple error page
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Visualization Error</title>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f5f7fa; 
                    padding: 2rem; 
                    color: #333;
                }}
                .error-container {{ 
                    max-width: 600px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 2rem; 
                    border-radius: 8px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    border-left: 4px solid #dc3545;
                }}
                h1 {{ color: #dc3545; margin-top: 0; }}
                .error-details {{ 
                    background: #f8f9fa; 
                    padding: 1rem; 
                    border-radius: 4px; 
                    margin-top: 1rem;
                    font-family: monospace;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>🚨 Visualization Generation Failed</h1>
                <p>An error occurred while generating the HTML visualization dashboard.</p>
                <div class="error-details">{str(e)}</div>
                <p style="margin-top: 1.5rem; color: #666; font-size: 14px;">
                    Please check the logs for more details or contact the pipeline maintainer.
                </p>
            </div>
        </body>
        </html>
        """

        return HTMLString(error_html)


@step
def generate_prompt_visualization(
    agent_prompts: Dict[str, str],
    run_metadata: Optional[Dict[str, Any]] = None,
) -> HTMLString:
    """
    Generate focused HTML visualization of agent prompts.

    This step creates a dedicated view for examining the prompts used
    by different agents in the LangGraph workflow, useful for prompt
    engineering and debugging.

    Args:
        agent_prompts: Dictionary of agent names to their prompts
        run_metadata: Optional metadata about the pipeline run

    Returns:
        HTMLString: HTML page showing all agent prompts
    """
    try:
        logger.info("Generating prompt-focused HTML visualization...")

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build prompt sections
        prompt_sections = ""
        for agent_name, prompt in agent_prompts.items():
            escaped_prompt = (
                prompt.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            prompt_sections += f"""
            <div class="prompt-card">
                <h2>🎯 {agent_name.replace('_', ' ').title()}</h2>
                <div class="prompt-content">
                    <pre><code>{escaped_prompt}</code></pre>
                </div>
                <div class="prompt-stats">
                    <span class="stat">Characters: {len(prompt)}</span>
                    <span class="stat">Lines: {len(prompt.splitlines())}</span>
                    <span class="stat">Words: ~{len(prompt.split())}</span>
                </div>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LLM Agent Prompts - Daily Summarization Pipeline</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #f5f7fa;
                    margin: 0;
                    padding: 1rem;
                    line-height: 1.6;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 1rem;
                }}
                
                .header {{
                    background: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    margin-bottom: 2rem;
                    border-bottom: 3px solid #7a3ef4;
                }}
                
                .header h1 {{
                    margin: 0 0 0.5rem 0;
                    color: #2c3e50;
                    font-size: 2rem;
                    font-weight: 500;
                }}
                
                .header .subtitle {{
                    color: #666;
                    font-size: 1rem;
                    margin: 0;
                }}
                
                .prompt-card {{
                    background: white;
                    border-radius: 8px;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    border-left: 4px solid #7a3ef4;
                }}
                
                .prompt-card h2 {{
                    color: #7a3ef4;
                    margin: 0 0 1rem 0;
                    font-size: 1.5rem;
                }}
                
                .prompt-content {{
                    background: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 6px;
                    padding: 1.5rem;
                    overflow-x: auto;
                    margin-bottom: 1rem;
                }}
                
                .prompt-content pre {{
                    margin: 0;
                    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                    font-size: 13px;
                    line-height: 1.4;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                
                .prompt-stats {{
                    display: flex;
                    gap: 1rem;
                    font-size: 0.875rem;
                    color: #666;
                }}
                
                .stat {{
                    background: #e9ecef;
                    padding: 0.25rem 0.75rem;
                    border-radius: 12px;
                    font-weight: 500;
                }}
                
                .timestamp {{
                    text-align: center;
                    color: #999;
                    font-size: 0.875rem;
                    margin-top: 2rem;
                    padding-top: 1rem;
                    border-top: 1px dashed #ddd;
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 0.5rem;
                    }}
                    
                    .header, .prompt-card {{
                        padding: 1rem;
                    }}
                    
                    .header h1 {{
                        font-size: 1.5rem;
                    }}
                    
                    .prompt-stats {{
                        flex-direction: column;
                        gap: 0.5rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 LLM Agent Prompts</h1>
                    <p class="subtitle">Daily Summarization Pipeline • Generated on {timestamp}</p>
                </div>
                
                {prompt_sections}
                
                <div class="timestamp">
                    Generated by ZenML LLM Daily Summarization Pipeline on {timestamp}
                </div>
            </div>
        </body>
        </html>
        """

        logger.info(
            f"Prompt visualization generated for {len(agent_prompts)} agents"
        )
        return HTMLString(html_content)

    except Exception as e:
        logger.error(f"Failed to generate prompt visualization: {e}")
        raise
