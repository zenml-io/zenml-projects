"""Step to run inference on test scenarios."""

import asyncio
from typing import Annotated, List

from environment.models import Scenario
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


def _run_inference_loop(
    inference_config: dict,
    test_scenarios: List[Scenario],
    db_path: str,
    judge_model: str,
) -> List[dict]:
    """Internal async inference loop."""
    import art
    from agent.rollout import rollout
    from art.langgraph import wrap_rollout
    from art.local import LocalBackend
    from environment.models import EmailScenario

    async def _async_inference():
        # Initialize model from checkpoint
        model = art.TrainableModel(
            name=inference_config["name"],
            project=inference_config["project"],
            base_model=inference_config["checkpoint_path"],
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
            path=inference_config["art_path"],
        )
        await model.register(backend)

        predictions = []

        for i, scenario in enumerate(test_scenarios):
            logger.info(
                f"Running inference on scenario {i+1}/{len(test_scenarios)}"
            )

            email_scenario = EmailScenario(step=0, scenario=scenario)

            try:
                traj = await wrap_rollout(model, rollout)(
                    model,
                    email_scenario,
                    db_path=db_path,
                    judge_model=judge_model,
                )

                predictions.append(
                    {
                        "scenario_id": scenario.id,
                        "question": scenario.question,
                        "expected_answer": scenario.answer,
                        "expected_message_ids": scenario.message_ids,
                        "predicted_answer": (
                            traj.final_answer.answer
                            if traj.final_answer
                            else None
                        ),
                        "predicted_source_ids": (
                            traj.final_answer.source_ids
                            if traj.final_answer
                            else []
                        ),
                        "correct": traj.metrics.get("correct", 0),
                    }
                )
            except Exception as e:
                logger.error(f"Error on scenario {scenario.id}: {e}")
                predictions.append(
                    {
                        "scenario_id": scenario.id,
                        "question": scenario.question,
                        "expected_answer": scenario.answer,
                        "expected_message_ids": scenario.message_ids,
                        "predicted_answer": None,
                        "predicted_source_ids": [],
                        "correct": 0,
                        "error": str(e),
                    }
                )

        return predictions

    return asyncio.run(_async_inference())


@step
def run_inference(
    inference_config: dict,
    test_scenarios: List[Scenario],
    db_path: str,
    judge_model: str = "openai/gpt-4.1",
) -> Annotated[List[dict], "predictions"]:
    """Run the trained agent on test scenarios.

    For each scenario, the agent:
    1. Searches the email database using its learned strategy
    2. Provides a final answer with source references
    3. Gets judged for correctness by the judge model

    Args:
        inference_config: Configuration from load_trained_model step.
        test_scenarios: List of test scenarios to evaluate.
        db_path: Path to the email database.
        judge_model: LiteLLM model ID for correctness judging.

    Returns:
        List of prediction dictionaries with results for each scenario.
    """
    logger.info(f"Running inference on {len(test_scenarios)} test scenarios")

    predictions = _run_inference_loop(
        inference_config=inference_config,
        test_scenarios=test_scenarios,
        db_path=db_path,
        judge_model=judge_model,
    )

    correct_count = sum(p["correct"] for p in predictions)
    logger.info(
        f"Inference complete: {correct_count}/{len(predictions)} correct"
    )

    return predictions
