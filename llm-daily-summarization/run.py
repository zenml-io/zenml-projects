"""
LLM Summarization Pipeline - Main Entry Point

This module defines and runs the daily chat summarization pipeline using ZenML.
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import click

from src.steps.data_ingestion import chat_data_ingestion_step
from src.steps.evaluation import evaluation_step
from src.steps.langgraph_processing import langgraph_agent_step
from src.steps.mock_data_ingestion import mock_chat_data_ingestion_step
from src.steps.output_distribution import output_distribution_step
from src.steps.preprocessing import text_preprocessing_step
from src.steps.trace_retrieval import retrieve_traces_step
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.logger import get_logger

logger = get_logger(__name__)

# Docker settings for containerized execution
docker_settings = DockerSettings(
    requirements="requirements.txt",
    environment={
        "ZENML_LOGGING_VERBOSITY": "DEBUG",
        "LANGFUSE_TRACING": "true"
    }
)


@pipeline(
    name="daily_chat_summarization",
    enable_cache=False,
    settings={"docker": docker_settings}
)
def daily_chat_summarization_pipeline(
    data_sources: list[str] = ["discord"],
    output_targets: list[str] = ["notion", "slack"],
    channels_config: Dict[str, List[str]] | None = None,
    model_config: Dict[str, Any] = {},
    use_mock_data: bool = True
) -> None:
    """Daily chat summarization pipeline using LangGraph agents and Vertex AI.
    
    Args:
        data_sources: List of data sources to process (discord, slack)
        output_targets: List of output destinations (notion, slack, github)
        channels_config: Dict mapping source to list of channels
        model_config: Configuration for the LLM model
        use_mock_data: If True, uses mock data; if False, uses real chat APIs
    """
    if model_config is None:
        model_config = {
            "model_name": "gemini-2.5-flash",
            "max_tokens": 4000,
            "temperature": 0.1
        }
    
    logger.info(f"Starting daily summarization pipeline at {datetime.now()}")
    logger.info(f"Data sources: {data_sources}")
    logger.info(f"Output targets: {output_targets}")
    
    # Step 1: Data ingestion from Discord/Slack
    if use_mock_data:
        raw_conversations = mock_chat_data_ingestion_step(data_sources=data_sources)
    else:
        raw_conversations = chat_data_ingestion_step(
            data_sources=data_sources,
            channels_config=channels_config
        )
    
    # Step 2: Text preprocessing and cleaning
    cleaned_data = text_preprocessing_step(raw_data=raw_conversations)
    
    # Step 3: LangGraph agent processing with Vertex AI
    summaries_and_tasks, _ = langgraph_agent_step(
        cleaned_data=cleaned_data,
        model_config=model_config
    )
    
    # Step 4: Output distribution to multiple targets
    delivery_results = output_distribution_step(
        processed_data=summaries_and_tasks,
        output_targets=output_targets
    )
    
    # Step 5: Evaluation and monitoring (with embedded visualization)
    evaluation_metrics, _ = evaluation_step(
        summaries_and_tasks=summaries_and_tasks,
        raw_conversations=raw_conversations,
        delivery_results=delivery_results
    )
    
    # Step 6: Retrieve and visualize Langfuse traces from the complete pipeline run
    traces_viz = retrieve_traces_step(
        processed_data=summaries_and_tasks,
        time_window_minutes=30  # Look for traces in the last 30 minutes
    )
    
    logger.info("Pipeline completed successfully with comprehensive observability")
    return evaluation_metrics


@click.command()
@click.option("--mock-data/--real-data", default=True, 
              help="Use mock data (default) or real chat APIs")
def main(mock_data: bool):
    """Main function to run the pipeline with default configuration."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Default configuration
    config = {
        "data_sources": ["discord"],  # Start with Discord only
        "output_targets": ["notion"],  # Start with Notion output
        "channels_config": {
            "discord": ["panagent-team"],
            "slack": []
        },
        "model_config": {
            "model_name": "gemini-2.5-flash",
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 0.95
        },
        "use_mock_data": mock_data
    }
    
    # Run the pipeline
    try:
        daily_chat_summarization_pipeline(**config)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()