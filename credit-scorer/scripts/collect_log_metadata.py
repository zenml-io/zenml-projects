#!/usr/bin/env python3
"""
Script to collect log metadata for EU AI Act Article 12 compliance.

This script retrieves log URIs from all ZenML pipelines and creates
the necessary directory structure and metadata files to satisfy
record keeping requirements without duplicating log content.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import (
    DEPLOYMENT_PIPELINE_NAME,
    FEATURE_ENGINEERING_PIPELINE_NAME,
    RELEASES_DIR,
    TRAINING_PIPELINE_NAME,
)
from src.utils.compliance.data_loader import ComplianceDataLoader
from zenml.client import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the script."""
    # Create the pipeline_logs directory required for Article 12 compliance
    pipeline_logs_dir = Path(RELEASES_DIR).parent / "pipeline_logs"
    pipeline_logs_dir.mkdir(exist_ok=True, parents=True)

    # Create a metadata file for all pipeline logs
    all_logs_metadata = {}

    # Process each pipeline
    pipelines = [
        FEATURE_ENGINEERING_PIPELINE_NAME,
        TRAINING_PIPELINE_NAME,
        DEPLOYMENT_PIPELINE_NAME,
    ]

    # Get the client
    client = Client()

    for pipeline_name in pipelines:
        try:
            # Get the pipeline
            pipeline = client.get_pipeline(pipeline_name)
            if not pipeline:
                logger.warning(f"Pipeline {pipeline_name} not found")
                continue

            # Process the latest run from each pipeline
            if pipeline.runs:
                run = pipeline.runs[-1]
                run_id = str(run.id)

                # Get log information
                log_info = ComplianceDataLoader.get_pipeline_log_paths(
                    pipeline_name, run_id
                )

                if log_info and "log_uri" in log_info:
                    # Add to metadata collection
                    all_logs_metadata[pipeline_name] = log_info

                    # Create symlink if log file exists
                    log_uri = log_info["log_uri"]
                    log_file_path = Path(log_uri)

                    if log_file_path.exists():
                        # Create a unique name for the symlink
                        symlink_dest = (
                            pipeline_logs_dir / f"{pipeline_name}_{run_id}.log"
                        )

                        # Create symlink or copy file
                        try:
                            if symlink_dest.exists():
                                symlink_dest.unlink()

                            symlink_dest.symlink_to(log_file_path)
                            logger.info(
                                f"Created symlink to log file at {symlink_dest}"
                            )
                        except Exception:
                            # If symlink fails, copy the file
                            import shutil

                            shutil.copy2(log_file_path, symlink_dest)
                            logger.info(f"Copied log file to {symlink_dest}")
                    else:
                        logger.warning(f"Log file not found at {log_uri}")
            else:
                logger.warning(f"No runs found for pipeline {pipeline_name}")

        except Exception as e:
            logger.error(f"Error processing pipeline {pipeline_name}: {e}")

    # Save the combined metadata file
    metadata_path = pipeline_logs_dir / "logs_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_logs_metadata, f, indent=2)

    logger.info(f"Log metadata saved to {metadata_path}")
    logger.info(
        f"Article 12 (Record Keeping) compliance directory created at {pipeline_logs_dir}"
    )

    # Return a count of collected logs
    return len(all_logs_metadata)


if __name__ == "__main__":
    count = main()
    logger.info(f"Collected metadata for {count} pipeline logs")
