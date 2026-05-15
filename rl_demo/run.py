"""
Entry point for running the RL environment sweep pipeline.

Usage:
    python run.py

Runs the dynamic RL training pipeline on PufferLib's `squared` env — a small
single-agent navigation task that produces a visible learning curve within a
few hundred thousand steps on CPU/MPS. Swap `env_names` for richer envs
(`ocean-target`, `ocean-cartpole`, `ocean-connect4`, ...) once the demo is
configured the way you want.
"""

import datetime

import torch
from pipelines import rl_environment_sweep
from zenml.integrations.wandb.flavors.wandb_experiment_tracker_flavor import (
    WandbExperimentTrackerSettings,
)
from zenml.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the RL environment sweep pipeline."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # One W&B group per pipeline invocation: every train_agent run gets
    # the same `group=` so the dashboard's Group view shows them as a
    # single coordinated sweep.
    wandb_group = f"rl_sweep_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    logger.info(
        f"Starting RL sweep pipeline (device: {device}, wandb group: {wandb_group})"
    )

    rl_environment_sweep.with_options(
        settings={
            "experiment_tracker.wandb": WandbExperimentTrackerSettings(
                settings={"run_group": wandb_group},
            ),
        }
    )(
        env_names=["ocean-squared"],
        # PufferLib's stock squared config trains at lr=0.05 — bracket that
        # so some runs learn and some over/undershoot (good demo signal).
        learning_rates=[1e-3, 5e-3, 2e-2, 5e-2, 1e-1],
        total_timesteps=1_000_000,
        device=device,
        client_id="acme-corp",
        project="rl-optimization",
        data_source="internal-simulation",
        domain="operations-research",
    )


if __name__ == "__main__":
    main()
