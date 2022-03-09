from zenml.steps import step, Output
from .configs import PreTrainingConfigs
from dqn.model import ReplayBuffer


@step
def replay_buffer(config: PreTrainingConfigs,) -> ReplayBuffer:
    """
    Create a ReplayBuffer object with the given configs
    
    :param config: PreTrainingConfigs
    :type config: PreTrainingConfigs
    :return: A replay buffer object.
    """
    replay_buffer = ReplayBuffer(
        size=config.MEM_SIZE,
        input_shape=config.INPUT_SHAPE,
        use_per=config.USE_PER,
    )
    return replay_buffer
