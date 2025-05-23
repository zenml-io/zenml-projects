import logging

import click
from logging_config import configure_logging
from pipelines.parallel_research_pipeline import (
    parallelized_deep_research_pipeline,
)
from utils.helper_functions import check_required_env_vars

logger = logging.getLogger(__name__)


@click.command(
    help="""
Deep Research Agent - ZenML Pipeline for Comprehensive Research

Run a deep research pipeline that:
1. Generates a structured report outline
2. Researches each topic with web searches and LLM analysis
3. Refines content through multiple reflection cycles
4. Produces a formatted, comprehensive research report

Examples:

  \b
  # Run with default configuration
  python run.py

  \b
  # Run with a custom pipeline configuration file
  python run.py --config configs/custom_pipeline.yaml
  
  \b
  # Override the research query
  python run.py --query "My research topic"
  
  \b
  # Run with a custom number of sub-questions
  python run.py --max-sub-questions 15
"""
)
@click.option(
    "--config",
    type=str,
    default="configs/enhanced_research.yaml",
    help="Path to the pipeline configuration YAML file",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run",
)
@click.option(
    "--log-file",
    type=str,
    default=None,
    help="Path to log file (if not provided, logs only go to console)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging",
)
@click.option(
    "--query",
    type=str,
    default=None,
    help="Research query (overrides the query in the config file)",
)
@click.option(
    "--max-sub-questions",
    type=int,
    default=10,
    help="Maximum number of sub-questions to process in parallel",
)
@click.option(
    "--require-approval",
    is_flag=True,
    default=False,
    help="Enable human-in-the-loop approval for additional searches",
)
@click.option(
    "--approval-timeout",
    type=int,
    default=3600,
    help="Timeout in seconds for human approval (default: 3600)",
)
def main(
    config: str = "configs/enhanced_research.yaml",
    no_cache: bool = False,
    log_file: str = None,
    debug: bool = False,
    query: str = None,
    max_sub_questions: int = 10,
    require_approval: bool = False,
    approval_timeout: int = 3600,
):
    """Run the deep research pipeline.

    Args:
        config: Path to the pipeline configuration YAML file
        no_cache: Disable caching for the pipeline run
        log_file: Path to log file
        debug: Enable debug logging
        query: Research query (overrides the query in the config file)
        max_sub_questions: Maximum number of sub-questions to process in parallel
        require_approval: Enable human-in-the-loop approval for additional searches
        approval_timeout: Timeout in seconds for human approval
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    configure_logging(level=log_level, log_file=log_file)

    # Check that required environment variables are present using the helper function
    required_vars = ["SAMBANOVA_API_KEY", "TAVILY_API_KEY"]
    missing_vars = check_required_env_vars(required_vars)

    if missing_vars:
        logger.error(
            f"The following required environment variables are not set: {', '.join(missing_vars)}"
        )
        logger.info("Please set them with:")
        for var in missing_vars:
            logger.info(f"  export {var}=your_{var.lower()}_here")
        return

    # Set pipeline options
    pipeline_options = {"config_path": config}

    if no_cache:
        pipeline_options["enable_cache"] = False

    logger.info("\n" + "=" * 80)
    logger.info("Starting Deep Research")
    logger.info("Using parallel pipeline for efficient execution")

    # Set up the pipeline with the parallelized version as default
    pipeline = parallelized_deep_research_pipeline.with_options(
        **pipeline_options
    )

    # Execute the pipeline
    if query:
        logger.info(
            f"Using query: {query} with max {max_sub_questions} parallel sub-questions"
        )
        if require_approval:
            logger.info(
                f"Human approval enabled with {approval_timeout}s timeout"
            )
        run = pipeline(
            query=query,
            max_sub_questions=max_sub_questions,
            require_approval=require_approval,
            approval_timeout=approval_timeout,
        )
    else:
        logger.info(
            f"Using query from config file with max {max_sub_questions} parallel sub-questions"
        )
        if require_approval:
            logger.info(
                f"Human approval enabled with {approval_timeout}s timeout"
            )
        run = pipeline(
            max_sub_questions=max_sub_questions,
            require_approval=require_approval,
            approval_timeout=approval_timeout,
        )

    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
