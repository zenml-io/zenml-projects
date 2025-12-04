"""Inference pipeline for deployment as HTTP service.

This pipeline is designed to be deployed via ZenML Pipeline Deployments,
exposing the trained email agent as an HTTP endpoint for real-time queries.
"""

from steps.inference import run_single_inference
from zenml import Model, pipeline
from zenml.config import DeploymentSettings

# Deployment settings for the HTTP service
deployment_settings = DeploymentSettings(
    app_title="ART Email Search Agent",
    app_description=(
        "Email search agent trained with OpenPipe ART + LangGraph. "
        "Answers questions about emails using a ReAct agent pattern."
    ),
    app_version="1.0.0",
    docs_url_path="/docs",
    invoke_url_path="/invoke",
    health_url_path="/health",
    cors={
        "allow_origins": ["*"],
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"],
    },
    uvicorn_host="0.0.0.0",
    uvicorn_port=8080,
)


@pipeline(
    model=Model(
        name="art-email-agent",
        description="Email search agent for inference",
        tags=["art", "langgraph", "inference", "deployment"],
    ),
    settings={"deployment": deployment_settings},
)
def inference_pipeline(
    question: str,
    inbox_address: str,
    query_date: str,
    checkpoint_path: str = "./.art/checkpoints/latest",
    db_path: str = "./enron_emails.db",
    judge_model: str = "openai/gpt-4.1",
    model_name: str = "art-email-agent",
    project_name: str = "email-search-agent",
    art_path: str = "./.art",
) -> dict:
    """Inference pipeline for single email search queries.

    This pipeline can be deployed as an HTTP service using ZenML's
    Pipeline Deployments feature. Each invocation runs a single query
    through the trained agent.

    Example invocation via HTTP:
        POST /invoke
        {
            "parameters": {
                "question": "What meeting is scheduled for next week?",
                "inbox_address": "john.smith@enron.com",
                "query_date": "2001-05-15"
            }
        }

    Args:
        question: The question to answer about the user's emails.
        inbox_address: The email address of the inbox to search.
        query_date: The reference date for the query (YYYY-MM-DD).
        checkpoint_path: Path to the trained model checkpoint.
        db_path: Path to the email database.
        judge_model: LiteLLM model ID for correctness judging.
        model_name: Name of the ART model.
        project_name: Name of the ART project.
        art_path: Path to ART artifacts directory.

    Returns:
        Dictionary with the agent's answer and metadata.
    """
    result = run_single_inference(
        question=question,
        inbox_address=inbox_address,
        query_date=query_date,
        checkpoint_path=checkpoint_path,
        db_path=db_path,
        judge_model=judge_model,
        model_name=model_name,
        project_name=project_name,
        art_path=art_path,
    )
    return result
