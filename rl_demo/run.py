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

import torch
from pipelines import rl_environment_sweep
from zenml.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the RL environment sweep pipeline."""
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    logger.info(f"Starting RL sweep pipeline (device: {device})")

    rl_environment_sweep(
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
