import os
import click
from zenml.artifacts.utils import load_artifact
from zenml.logger import get_logger
import json
from zenml.client import Client
from pipelines.data_processing_pipeline import document_processing_pipeline
from pipelines.agent_analysis_pipeline import agent_analysis_pipeline
from pipelines.agent_validation_pipeline import agent_validation_pipeline
from pipelines.report_dashboard_pipeline import report_dashboard_pipeline
from utils import string_to_dict

logger = get_logger(__name__)


@click.command(
    help="""
ZenML Financial Report Processing Pipeline.

Run the pipeline with the specified file path.

Examples:

  \b
  # Run the pipeline with a specific CSV file
    python run.py --file_path data/financial_reports.csv
"""
)
@click.option(
    "--file_path",
    type=str,
    required=True,
    help="Path to the financial reports CSV file.",
)
def main(file_path: str):
    """Main entry point for the pipeline execution."""
    logger.info(f"Running pipeline with file: {file_path}")
    document_processing_pipeline(file_path=file_path)
    
    company_data = Client().get_artifact_version("structured_dataset").load()
    
    logger.info("Running Agent Analysis of the company's data")
    print(company_data["0"])
    agent_analysis_pipeline(company_data["0"])

    metric_result = Client().get_artifact_version("metric_result").load()
    context_result = Client().get_artifact_version("context_result").load()
    competitor_result = (
        Client().get_artifact_version("competitor_result").load()
    )
    risk_result = Client().get_artifact_version("risk_assesment_result").load()
    strategy_result = (
        Client().get_artifact_version("strategic_direction_result").load()
    )
    response_data = {
        "metric_result": metric_result,
        "context_result": context_result,
        "competitor_result": competitor_result,
        "risk_assesment_result": risk_result,
        "strategic_direction_result": strategy_result,
    }
    logger.info("Running Agent Validation on the company's data")
    agent_validation_pipeline(response_data, company_data["0"])

    logger.info("Generating Report")
    synthesis_report = Client().get_artifact_version("synthesis_result").load()
    synthesis_report = string_to_dict(synthesis_report)

    logger.info("Reporting on the Dashboard")
    report_dashboard_pipeline(synthesis_report)


if __name__ == "__main__":
    main()
