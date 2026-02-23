"""
Entry point for running the RL environment sweep pipeline.

Usage:
    python run.py

Runs the dynamic RL training pipeline on Connect4 with 5 learning rates.
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
        env_names=["ocean-connect4"],
        learning_rates=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        total_timesteps=100_000,
        device=device,
        client_id="acme-corp",
        project="rl-optimization",
        data_source="internal-simulation",
        domain="operations-research",
    )


if __name__ == "__main__":
    main()
