"""Step to run a single inference query."""

import asyncio
from typing import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


def _run_single_query(
    question: str,
    inbox_address: str,
    query_date: str,
    checkpoint_path: str,
    db_path: str,
    judge_model: str,
    model_name: str,
    project_name: str,
    art_path: str,
) -> dict:
    """Internal async inference for a single query."""
    import art
    from agent.rollout import rollout
    from art.langgraph import wrap_rollout
    from art.local import LocalBackend
    from environment.models import EmailScenario, Scenario

    async def _async_query():
        # Initialize model from checkpoint
        model = art.TrainableModel(
            name=model_name,
            project=project_name,
            base_model=checkpoint_path,
        )

        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(max_seq_length=8192),
            engine_args=art.dev.EngineArgs(
                enforce_eager=True,
                gpu_memory_utilization=0.8,
            ),
        )

        backend = LocalBackend(
            in_process=True,
            path=art_path,
        )
        await model.register(backend)

        # Create a scenario from the query parameters
        scenario = Scenario(
            id=0,
            question=question,
            answer="",  # Unknown - this is what we're trying to find
            message_ids=[],
            how_realistic=1.0,
            inbox_address=inbox_address,
            query_date=query_date,
            split="inference",
        )

        email_scenario = EmailScenario(step=0, scenario=scenario)

        try:
            traj = await wrap_rollout(model, rollout)(
                model,
                email_scenario,
                db_path=db_path,
                judge_model=judge_model,
            )

            return {
                "question": question,
                "inbox_address": inbox_address,
                "query_date": query_date,
                "answer": (
                    traj.final_answer.answer if traj.final_answer else None
                ),
                "source_ids": (
                    traj.final_answer.source_ids if traj.final_answer else []
                ),
                "success": traj.final_answer is not None,
            }
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return {
                "question": question,
                "inbox_address": inbox_address,
                "query_date": query_date,
                "answer": None,
                "source_ids": [],
                "success": False,
                "error": str(e),
            }

    return asyncio.run(_async_query())


@step
def run_single_inference(
    question: str,
    inbox_address: str,
    query_date: str,
    checkpoint_path: str = "./.art/checkpoints/latest",
    db_path: str = "./enron_emails.db",
    judge_model: str = "openai/gpt-4.1",
    model_name: str = "art-email-agent",
    project_name: str = "email-search-agent",
    art_path: str = "./.art",
) -> Annotated[dict, "inference_result"]:
    """Run a single inference query through the trained agent.

    This step loads the trained model and runs a single email search
    query, returning the agent's answer with source references.

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
    logger.info(f"Running inference for question: {question[:50]}...")

    result = _run_single_query(
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

    if result.get("success"):
        logger.info(f"Inference successful: {result['answer'][:100]}...")
    else:
        logger.warning(f"Inference failed: {result.get('error', 'No answer')}")

    return result
