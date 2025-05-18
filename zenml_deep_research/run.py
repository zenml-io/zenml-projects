import os
import click
import logging
from pipelines.research_pipeline import enhanced_deep_research_pipeline
from logging_config import configure_logging

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
def main(
    config: str = "configs/enhanced_research.yaml",
    no_cache: bool = False,
    log_file: str = None,
    debug: bool = False,
    query: str = None,
):
    """Run the deep research pipeline.

    Args:
        config: Path to the pipeline configuration YAML file
        no_cache: Disable caching for the pipeline run
        log_file: Path to log file
        debug: Enable debug logging
        query: Research query (overrides the query in the config file)
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    configure_logging(level=log_level, log_file=log_file)

    # Check that required environment variables are present
    missing_vars = []

    if not os.environ.get("SAMBANOVA_API_KEY"):
        missing_vars.append("SAMBANOVA_API_KEY")

    if not os.environ.get("TAVILY_API_KEY"):
        missing_vars.append("TAVILY_API_KEY")

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

    # Set up the pipeline
    pipeline = enhanced_deep_research_pipeline.with_options(**pipeline_options)

    # Execute the pipeline
    if query:
        logger.info(f"Using query: {query}")
        run = pipeline(query=query)
    else:
        logger.info("Using query from config file")
        run = pipeline()

    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
