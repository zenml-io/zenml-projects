from zenml.steps import step, Output
from .configs import PreTrainingConfigs
from dqn.model import ReplayBuffer


@step
def replay_buffer(config: PreTrainingConfigs,) -> ReplayBuffer:
    """
    Create a ReplayBuffer object with the given configs
    Args:
        config: PreTrainingConfigs
    """
    replay_buffer = ReplayBuffer(
        size=config.MEM_SIZE,
        input_shape=config.INPUT_SHAPE,
        use_per=config.USE_PER,
    )
    return replay_buffer
