"""Experiment tracking helpers for RL pipeline steps."""

from zenml.client import Client


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
