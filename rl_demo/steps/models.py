"""Pydantic models for RL pipeline (avoids ZenML/Pydantic dataclass Field conflict)."""

from pydantic import BaseModel

# PolicyCheckpoint: dict with model_state_dict, config, metrics_history
PolicyCheckpoint = dict


class EnvConfig(BaseModel):
    """Configuration for a single RL environment + hyperparameter combo."""

    env_name: str
    num_envs: int = 64
    num_workers: int = 1
    learning_rate: float = 3e-4
    batch_size: int = 8192
    total_timesteps: int = 100_000
    update_epochs: int = 3
    device: str = "cuda"
    tag: str = ""


class TrainingResult(BaseModel):
    """Output of a single training run — versioned as a ZenML artifact."""

    env_name: str
    tag: str
    mean_reward: float
    mean_episode_length: float
    total_timesteps: int
    steps_per_second: float
    policy_loss: float
    value_loss: float
    entropy: float
    config: dict
    metrics_history: list[dict] = []


class EvalResult(BaseModel):
    """Evaluation result for a trained policy."""

    env_name: str
    tag: str
    eval_mean_reward: float
    eval_std_reward: float
    eval_episodes: int
    is_best: bool = False


class DatasetMetadata(BaseModel):
    """
    Metadata for a versioned training dataset.

    Every training run traces back to this artifact — so when legal says
    "delete all data for Client X", you can find every model version
    that touched that data via the ZenML lineage graph.
    """

    client_id: str
    project: str
    data_source: str
    domain: str
    env_names: list[str]
    description: str = ""
