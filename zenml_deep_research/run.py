import logging

import click
from logging_config import configure_logging
from pipelines.parallel_research_pipeline import (
    parallelized_deep_research_pipeline,
)
from utils.helper_functions import check_required_env_vars

logger = logging.getLogger(__name__)

# Research mode presets for easy configuration
RESEARCH_MODES = {
    "rapid": {
        "max_sub_questions": 5,
        "num_results_per_search": 2,
        "max_additional_searches": 0,
        "description": "Quick research with minimal depth - great for getting a fast overview",
    },
    "balanced": {
        "max_sub_questions": 10,
        "num_results_per_search": 3,
        "max_additional_searches": 2,
        "description": "Balanced research with moderate depth - ideal for most use cases",
    },
    "deep": {
        "max_sub_questions": 15,
        "num_results_per_search": 5,
        "max_additional_searches": 4,
        "description": "Comprehensive research with maximum depth - for thorough analysis",
        "suggest_approval": True,  # Suggest using approval for deep mode
    },
}


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
  # Use a research mode preset for easy configuration
  python run.py --mode rapid    # Quick overview
  python run.py --mode balanced  # Standard research (default)
  python run.py --mode deep      # Comprehensive analysis

  \b
  # Run with a custom pipeline configuration file
  python run.py --config configs/custom_pipeline.yaml
  
  \b
  # Override the research query
  python run.py --query "My research topic"
  
  \b
  # Combine mode with other options
  python run.py --mode deep --query "Complex topic" --require-approval
  
  \b
  # Run with a custom number of sub-questions
  python run.py --max-sub-questions 15
"""
)
@click.option(
    "--mode",
    type=click.Choice(["rapid", "balanced", "deep"], case_sensitive=False),
    default=None,
    help="Research mode preset: rapid (fast overview), balanced (standard), or deep (comprehensive)",
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
    mode: str = None,
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
        mode: Research mode preset (rapid, balanced, or deep)
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

    # Apply mode presets if specified
    if mode:
        mode_config = RESEARCH_MODES[mode.lower()]
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Using research mode: {mode.upper()}")
        logger.info(f"Description: {mode_config['description']}")

        # Apply mode parameters (can be overridden by explicit arguments)
        if max_sub_questions == 10:  # Default value - apply mode preset
            max_sub_questions = mode_config["max_sub_questions"]
            logger.info(f"  - Max sub-questions: {max_sub_questions}")

        # Store mode config for later use
        mode_max_additional_searches = mode_config["max_additional_searches"]
        mode_num_results_per_search = mode_config["num_results_per_search"]

        logger.info(
            f"  - Max additional searches: {mode_max_additional_searches}"
        )
        logger.info(
            f"  - Results per search: {mode_num_results_per_search} (configure in YAML for step-level control)"
        )

        # Check if a mode-specific config exists and user didn't override config
        if config == "configs/enhanced_research.yaml":  # Default config
            mode_specific_config = f"configs/{mode.lower()}_research.yaml"
            import os

            if os.path.exists(mode_specific_config):
                config = mode_specific_config
                logger.info(f"  - Using mode-specific config: {config}")

        # Suggest approval for deep mode if not already enabled
        if mode_config.get("suggest_approval") and not require_approval:
            logger.info(f"\n{'!' * 60}")
            logger.info(
                f"! TIP: Consider using --require-approval with {mode} mode"
            )
            logger.info(f"! for better control over comprehensive research")
            logger.info(f"{'!' * 60}")

        logger.info(f"{'=' * 80}\n")
    else:
        # Default values if no mode specified
        mode_max_additional_searches = 2
        mode_num_results_per_search = 3

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
            max_additional_searches=mode_max_additional_searches,
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
            max_additional_searches=mode_max_additional_searches,
        )

    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
