from sklearn.base import RegressorMixin
from zenml.post_execution import get_pipeline
from zenml.steps import StepContext, step
from zenml.steps.step_output import Output


@step
def model_picker(
    context: StepContext,
) -> Output(model=RegressorMixin, associated_run_id=str):
    """Picks the best models from all previous training pipeline runs.

    Args:
        context: Step context to access previous runs

    Returns:
        model: The best model based on previous metrics.
        associated_run_id: The associated run ID of the training pipeline that
        produced that mode.
    """
    training_pipeline = get_pipeline(pipeline_name="training_pipeline")
    last_run = training_pipeline.runs[0]

    best_score = None
    best_model = None
    best_run = None
    model = None
    mae = None

    try:
        mae = last_run.get_step(name="tester").output.read()
        print(f"Run {last_run.name} yielded a model with mae={mae}")
    except KeyError:
        print(
            f"Skipping {last_run.name} as it does not contain the tester step"
        )

    try:
        model = last_run.get_step(name="trainer").output.read()
    except KeyError:
        print(
            f"Skipping {last_run.name} as it does not contain the trainer step"
        )

    if mae and model:
        if not best_score or (best_score and mae <= best_score):
            best_model = model
            best_score = mae
            best_run = last_run.name

    print(
        f"Choosing model from pipeline run: {best_run} with mae of "
        f"{best_score}"
    )

    return best_model, best_run
