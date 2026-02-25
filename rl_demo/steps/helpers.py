"""PufferLib env setup and policy (shared by train + eval steps)."""

import sys

import torch
import torch.nn as nn

import pufferlib.pufferl as pufferl
from pufferlib.pytorch import layer_init

from steps.models import PolicyCheckpoint


def resolve_device(requested: str) -> str:
    """Resolve requested device to one available in the current environment.

    MPS is Apple-only; CUDA requires GPU. When running in K8s/Linux containers,
    'mps' from a Mac client will fail — fall back to cpu.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps":
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except (AttributeError, RuntimeError):
            pass
    return "cpu"


def puffer_env_name(name: str) -> str:
    """Map 'ocean-pong' → 'puffer_pong' for PufferLib config lookup."""
    if name.startswith("ocean-"):
        return "puffer_" + name[6:]
    return name


def make_vecenv(env_name: str, **overrides):
    """Create a PufferLib vectorized env with config overrides."""
    puffer_name = puffer_env_name(env_name)
    # PufferLib's load_config() parses sys.argv; ZenML injects
    # --entrypoint_config_source, --snapshot_id, --run_id which PufferLib rejects.
    # Clear argv so parse_args() uses config defaults; we override via overrides below.
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        args = pufferl.load_config(puffer_name)
    finally:
        sys.argv = _argv
    for section, updates in overrides.items():
        args[section].update(updates)
    return pufferl.load_env(puffer_name, args), puffer_name, args


def make_policy(
    vecenv,
    device: str = "cpu",
    checkpoint: PolicyCheckpoint | None = None,
):
    """Create an RLPolicy from vecenv spaces, optionally loading a checkpoint."""
    obs_dim = vecenv.single_observation_space.shape[0]
    act_dim = vecenv.single_action_space.n
    policy = RLPolicy(obs_dim, act_dim)
    if checkpoint:
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
    else:
        policy = policy.to(device)
    return policy


def extract_logs(logs: dict | None) -> dict:
    """Pull standard metrics from PuffeRL log dict."""
    if logs is None:
        return {
            "mean_reward": 0,
            "mean_episode_length": 0,
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "sps": 0,
        }

    def _get(key, alt=None):
        return logs.get(key, logs.get(alt or key, 0))

    return {
        "mean_reward": _get("environment/mean_reward", "environment/score"),
        "mean_episode_length": _get(
            "environment/mean_episode_length", "environment/episode_length"
        ),
        "policy_loss": _get("losses/policy_loss"),
        "value_loss": _get("losses/value_loss"),
        "entropy": _get("losses/entropy"),
        "sps": _get("SPS"),
    }


def run_eval_episodes(policy, vecenv, num_episodes: int) -> list[float]:
    """Run evaluation episodes and return per-episode rewards."""
    rewards = []
    obs, _ = vecenv.reset()
    episode_rewards = [0.0] * vecenv.num_agents
    completed = 0
    while completed < num_episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(torch.FloatTensor(obs))
            actions = logits.argmax(dim=-1).numpy()
        obs, reward, terminated, truncated, _ = vecenv.step(actions)
        for i in range(vecenv.num_agents):
            episode_rewards[i] += float(reward[i])
            if terminated[i] or truncated[i]:
                rewards.append(episode_rewards[i])
                episode_rewards[i] = 0.0
                completed += 1
    return rewards


class RLPolicy(nn.Module):
    """
    Simple MLP policy compatible with PufferLib's PuffeRL trainer.
    Swap this out for CNN, LSTM, Transformer — whatever your research needs.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, observations, state=None):
        features = self.encoder(observations)
        return self.actor(features), self.critic(features)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)
