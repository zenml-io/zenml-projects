"""Experiment tracking helpers for RL pipeline steps."""

import wandb
from zenml.client import Client
from zenml.steps import get_step_context


def active_wandb_tracker_name() -> str:
    """Return the active stack's W&B experiment tracker name."""
    tracker = Client().active_stack.experiment_tracker
    if tracker is None:
        raise RuntimeError(
            "The active ZenML stack has no experiment tracker. Attach your "
            "W&B tracker with `zenml stack update <stack-name> -e <tracker-name>`."
        )
    if getattr(tracker, "flavor", None) != "wandb":
        raise RuntimeError(
            "The active ZenML experiment tracker must use the `wandb` flavor "
            "because this pipeline logs metrics with the W&B SDK."
        )
    return tracker.name


def log_zenml_context_to_wandb() -> dict[str, str]:
    """Attach ZenML run context to the active W&B run."""
    context = get_step_context()

    metadata = {
        "zenml_pipeline_run_id": str(context.pipeline_run.id),
        "zenml_pipeline_run_name": context.pipeline_run.name,
        "zenml_step_name": context.step_name,
        "zenml_pipeline_name": context.pipeline.name,
    }

    wandb.config.update(metadata, allow_val_change=True)
    wandb.summary.update(metadata)
    if wandb.run:
        wandb.run.tags += (
            f"zenml_pipeline:{context.pipeline.name}",
            f"zenml_run:{context.pipeline_run.id}",
            f"zenml_step:{context.step_name}",
        )
    return metadata
