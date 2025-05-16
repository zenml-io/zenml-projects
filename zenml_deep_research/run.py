import os
import click
import logging
from pipelines.research_pipeline import deep_research_pipeline
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
  # Run with a simple query
  python run.py --query "Explain the impact of quantum computing on cryptography"

  \b
  # Run with more reflection cycles
  python run.py --query "History of artificial intelligence" --num-reflections 3
  
  \b
  # Run with a custom pipeline configuration file
  python run.py --query "Climate change adaptation strategies" --config configs/custom_pipeline.yaml
  
  \b
  # Save the report to a file
  python run.py --query "Climate change adaptation strategies" --output report.html
"""
)
@click.option(
    "--query",
    type=str,
    required=True,
    help="The research query or topic to investigate",
)
@click.option(
    "--config",
    type=str,
    default="configs/pipeline_config.yaml",
    help="Path to the pipeline configuration YAML file",
)
@click.option(
    "--num-reflections",
    type=int,
    default=2,
    help="Number of reflection cycles to perform for each paragraph",
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
def main(
    query: str,
    config: str = "configs/pipeline_config.yaml",
    num_reflections: int = 2,
    no_cache: bool = False,
    log_file: str = None,
    debug: bool = False,
):
    """Run the deep research pipeline.

    Args:
        query: The research query or topic to investigate
        config: Path to the pipeline configuration YAML file
        output: Path to save the output HTML report
        num_reflections: Number of reflection cycles to perform
        no_cache: Disable caching for the pipeline run
        log_file: Path to log file
        debug: Enable debug logging
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
    logger.info(f"Starting Deep Research on query: {query}")
    logger.info(f"Number of reflection cycles: {num_reflections}")
    logger.info(f"Using pipeline configuration from: {config}")
    logger.info("=" * 80 + "\n")

    # Run the pipeline
    # Step parameters are passed directly, while pipeline config is loaded from the YAML
    run = deep_research_pipeline.with_options(**pipeline_options)(query=query)

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
