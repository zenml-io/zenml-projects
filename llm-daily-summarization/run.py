"""
LLM Summarization Pipeline - Main Entry Point

This module defines and runs the daily chat summarization pipeline using ZenML.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
from src.steps.data_ingestion import chat_data_ingestion_step
from src.steps.evaluation import evaluation_step
from src.steps.langgraph_processing import langgraph_agent_step
from src.steps.mock_data_ingestion import mock_chat_data_ingestion_step
from src.steps.output_distribution import output_distribution_step
from src.steps.trace_retrieval import retrieve_traces_step
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.logger import get_logger

logger = get_logger(__name__)

# Docker settings for containerized execution
docker_settings = DockerSettings(
    requirements="requirements.txt",
    environment={
        "ZENML_LOGGING_VERBOSITY": "DEBUG",
        "LANGFUSE_TRACING": "true",
    },
)


@pipeline(
    name="daily_chat_summarization",
    enable_cache=False,
    settings={"docker": docker_settings},
)
def daily_chat_summarization_pipeline(
    data_sources: list[str] = ["discord"],
    output_targets: list[str] = ["notion", "slack"],
    channels_config: Dict[str, List[str]] | None = None,
    model_config: Dict[str, Any] = {},
    use_mock_data: bool = True,
    days_back: int = 1,
    max_messages: Optional[int] = None,
    include_threads: bool = True,
    extract_tasks: bool = True,  # NEW: controls whether tasks are extracted/delivered
) -> None:
    """Daily chat summarization pipeline orchestrating ingestion, processing,
    summarization, distribution and evaluation of chat conversations.

    Args:
        data_sources: List of chat platforms to ingest from
            (e.g. ``["discord", "slack"]``).
        output_targets: Destinations for the generated summaries
            (e.g. ``["notion", "slack", "github"]``).
        channels_config: Mapping from each source to the channels that should
            be fetched. If ``None`` a sensible default is applied.
        model_config: Configuration dictionary for the LLM agent
            (model name, temperature, max tokens, etc.).
        use_mock_data: When ``True`` the pipeline runs against a local JSON
            fixture instead of calling live APIs.
        days_back: The number of whole days of history to retrieve for each
            channel when **live** data is used. A value of ``1`` means "today
            and yesterday".
        max_messages: Upper limit on the number of messages to fetch per
            channel or thread. ``None`` disables the limit. This only applies
            when ``use_mock_data`` is ``False``.
        include_threads: When ``True`` the Discord ingestion logic will also
            collect thread messages in addition to root channel messages.
            Ignored when ``use_mock_data`` is ``True``.
        extract_tasks: When ``True``, task extraction and delivery are enabled
            in the LangGraph agent and output distribution steps.

    Returns:
        None. The pipeline writes artefacts to the ZenML stack and may return
        evaluation metrics for further inspection.
    """
    if model_config is None:
        model_config = {
            "model_name": "gemini-2.5-flash",
            "max_tokens": 4000,
            "temperature": 0.1,
        }

    logger.info(f"Starting daily summarization pipeline at {datetime.now()}")
    logger.info(f"Data sources: {data_sources}")
    logger.info(f"Output targets: {output_targets}")

    # Step 1: Data ingestion from Discord/Slack
    if use_mock_data:
        logger.info(
            "Using mock data - ignoring days_back, max_messages, and include_threads parameters"
        )
        raw_conversations = mock_chat_data_ingestion_step(
            data_sources=data_sources
        )
    else:
        raw_conversations = chat_data_ingestion_step(
            data_sources=data_sources,
            channels_config=channels_config,
            days_back=days_back,
            max_messages=max_messages,
            include_threads=include_threads,
        )

    # Step 2: LangGraph agent processing with Vertex AI
    summaries_and_tasks, _ = langgraph_agent_step(
        raw_data=raw_conversations,
        model_config=model_config,
        extract_tasks=extract_tasks,
    )

    # Step 3: Output distribution to multiple targets
    delivery_results = output_distribution_step(
        processed_data=summaries_and_tasks,
        output_targets=output_targets,
        extract_tasks=extract_tasks,  # NEW
    )

    # Step 4: Evaluation and monitoring (with embedded visualization)
    evaluation_metrics, _ = evaluation_step(
        summaries_and_tasks=summaries_and_tasks,
        raw_conversations=raw_conversations,
        delivery_results=delivery_results,
    )

    # Step 5: Retrieve and visualize Langfuse traces from the complete pipeline run
    traces_viz = retrieve_traces_step(
        processed_data=summaries_and_tasks,
        time_window_minutes=30,  # Look for traces in the last 30 minutes
    )

    logger.info(
        "Pipeline completed successfully with comprehensive observability"
    )
    return evaluation_metrics


@click.command()
@click.option(
    "--mock-data/--real-data",
    default=True,
    help="Use mock data (default) or real chat APIs",
)
@click.option(
    "--days-back",
    default=1,
    show_default=True,
    type=int,
    help="Number of days to look back when fetching messages",
)
@click.option(
    "--max-messages",
    default=None,
    type=int,
    help="Maximum messages to fetch per channel/thread (None for unlimited)",
)
@click.option(
    "--include-threads/--no-threads",
    default=True,
    show_default=True,
    help="Include thread messages when fetching Discord data",
)
@click.option(
    "--task-list/--no-task-list",
    default=True,
    show_default=True,
    help="Enable or disable task extraction and delivery",
)
@click.option(
    "--output-targets",
    multiple=True,
    default=["notion"],
    show_default=True,
    help="Output targets for summaries (notion, slack, discord, local)",
)
@click.option(
    "--discord-channel-id",
    default=None,
    help="Discord channel ID for posting summaries (overrides env var)",
)
def main(
    mock_data: bool,
    days_back: int,
    max_messages: Optional[int],
    include_threads: bool,
    task_list: bool,
    output_targets: tuple,
    discord_channel_id: Optional[str],
):
    """Main function to run the pipeline with default configuration."""
    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Override Discord channel ID if provided via CLI
    if discord_channel_id:
        os.environ["DISCORD_SUMMARY_CHANNEL_ID"] = discord_channel_id

    # Default configuration
    config = {
        "data_sources": ["discord"],  # Start with Discord only
        "output_targets": list(output_targets)
        if output_targets
        else ["notion"],
        "channels_config": {
            "discord": [
                "panagent-team",
                "growth-team",
                "product-team",
                "marketing-team",
                "internal-team",
                "help-me",
                "random",
            ],
            "slack": [],
        },
        "model_config": {
            "model_name": "gemini-2.5-flash",
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 0.95,
        },
        "use_mock_data": mock_data,
        "days_back": days_back,
        "max_messages": max_messages,
        "include_threads": include_threads,
        "extract_tasks": task_list,  # NEW
    }

    # Run the pipeline
    try:
        daily_chat_summarization_pipeline(**config)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
