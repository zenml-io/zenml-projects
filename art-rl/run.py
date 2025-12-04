#!/usr/bin/env python
"""CLI entry point for the ART Email Search Agent project.

This script provides commands to run the different pipelines:
- data: Prepare data artifacts (download emails, create database, load scenarios)
- train: Train the agent using ART with GRPO and RULER
- eval: Evaluate a trained model on test scenarios
- all: Run the complete workflow (data → train → eval)

Examples:
    # Prepare data (run once, artifacts are cached)
    python run.py --pipeline data

    # Train with local GPU
    python run.py --pipeline train --config configs/training_local.yaml

    # Train on Kubernetes
    python run.py --pipeline train --config configs/training_k8s.yaml

    # Evaluate trained model
    python run.py --pipeline eval

    # Run complete workflow
    python run.py --pipeline all
"""

import argparse
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def check_environment():
    """Verify required environment variables are set."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Required for RULER scoring.")
        print("Set it in .env file or export OPENAI_API_KEY=your-key")

    if not os.environ.get("WANDB_API_KEY"):
        print(
            "INFO: WANDB_API_KEY not set. Weights & Biases logging disabled."
        )


def run_data_pipeline(config_path: Optional[str] = None):
    """Run the data preparation pipeline."""
    from pipelines.data_preparation import data_preparation_pipeline

    print("=" * 60)
    print("Running Data Preparation Pipeline")
    print("=" * 60)

    if config_path:
        data_preparation_pipeline.with_options(config_path=config_path)()
    else:
        data_preparation_pipeline()

    print("\nData preparation complete!")
    print("Artifacts are cached and will be reused in subsequent runs.")


def run_training_pipeline(
    config_path: Optional[str] = None,
    no_cache: bool = False,
):
    """Run the training pipeline."""
    from pipelines.training import training_pipeline
    from zenml.client import Client

    print("=" * 60)
    print("Running Training Pipeline")
    print("=" * 60)

    # Get artifacts from most recent data preparation run
    client = Client()

    try:
        # Find the most recent data preparation pipeline run
        runs = client.list_pipeline_runs(
            pipeline_name_or_id="data_preparation_pipeline",
            sort_by="desc:created",
            size=1,
        )

        if not runs.items:
            print("ERROR: No data preparation run found.")
            print("Please run: python run.py --pipeline data")
            return

        latest_run = runs.items[0]
        print(f"Using data from run: {latest_run.id}")

        # Get artifacts from the run
        db_path = latest_run.steps["create_database"].output.load()
        train_scenarios = (
            latest_run.steps["load_scenarios"]
            .outputs["train_scenarios"]
            .load()
        )

        print(f"Loaded {len(train_scenarios)} training scenarios")
        print(f"Database path: {db_path}")

    except Exception as e:
        print(f"ERROR: Failed to load data artifacts: {e}")
        print("Please run: python run.py --pipeline data")
        return

    # Run training
    pipeline_instance = training_pipeline

    if config_path:
        pipeline_instance = pipeline_instance.with_options(
            config_path=config_path
        )

    if no_cache:
        pipeline_instance = pipeline_instance.with_options(enable_cache=False)

    pipeline_instance(
        train_scenarios=train_scenarios,
        db_path=db_path,
    )

    print("\nTraining complete!")


def run_evaluation_pipeline(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
):
    """Run the evaluation pipeline."""
    from pipelines.evaluation import evaluation_pipeline
    from zenml.client import Client

    print("=" * 60)
    print("Running Evaluation Pipeline")
    print("=" * 60)

    client = Client()

    # Get test scenarios from data preparation
    try:
        data_runs = client.list_pipeline_runs(
            pipeline_name_or_id="data_preparation_pipeline",
            sort_by="desc:created",
            size=1,
        )

        if not data_runs.items:
            print("ERROR: No data preparation run found.")
            print("Please run: python run.py --pipeline data")
            return

        data_run = data_runs.items[0]
        db_path = data_run.steps["create_database"].output.load()
        test_scenarios = (
            data_run.steps["load_scenarios"].outputs["test_scenarios"].load()
        )

        print(f"Loaded {len(test_scenarios)} test scenarios")

    except Exception as e:
        print(f"ERROR: Failed to load data artifacts: {e}")
        return

    # Get checkpoint from training
    if not checkpoint_path:
        try:
            train_runs = client.list_pipeline_runs(
                pipeline_name_or_id="training_pipeline",
                sort_by="desc:created",
                size=1,
            )

            if not train_runs.items:
                print("ERROR: No training run found.")
                print("Please run: python run.py --pipeline train")
                return

            train_run = train_runs.items[0]
            checkpoint_path = (
                train_run.steps["train_agent"]
                .outputs["checkpoint_path"]
                .load()
            )
            model_config = train_run.steps["setup_art_model"].output.load()

            print(f"Using checkpoint: {checkpoint_path}")

        except Exception as e:
            print(f"ERROR: Failed to load training artifacts: {e}")
            return
    else:
        # Use default model config if checkpoint provided manually
        model_config = {
            "name": "art-email-agent",
            "project": "email-search-agent",
            "base_model": checkpoint_path,
        }

    # Run evaluation
    pipeline_instance = evaluation_pipeline

    if config_path:
        pipeline_instance = pipeline_instance.with_options(
            config_path=config_path
        )

    pipeline_instance(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        test_scenarios=test_scenarios,
        db_path=db_path,
    )

    print("\nEvaluation complete!")


def run_all_pipelines(config_path: Optional[str] = None):
    """Run the complete workflow: data → train → eval."""
    print("=" * 60)
    print("Running Complete Workflow")
    print("=" * 60)

    run_data_pipeline()
    run_training_pipeline(config_path=config_path)
    run_evaluation_pipeline()

    print("\n" + "=" * 60)
    print("Complete workflow finished!")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ART Email Search Agent - ZenML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--pipeline",
        "-p",
        choices=["data", "train", "eval", "all"],
        required=True,
        help="Pipeline to run",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable ZenML caching",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (for eval pipeline)",
    )

    args = parser.parse_args()

    # Check environment
    check_environment()

    # Run the selected pipeline
    if args.pipeline == "data":
        run_data_pipeline(config_path=args.config)

    elif args.pipeline == "train":
        run_training_pipeline(
            config_path=args.config,
            no_cache=args.no_cache,
        )

    elif args.pipeline == "eval":
        run_evaluation_pipeline(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
        )

    elif args.pipeline == "all":
        run_all_pipelines(config_path=args.config)


if __name__ == "__main__":
    main()
