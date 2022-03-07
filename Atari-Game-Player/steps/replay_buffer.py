from zenml.steps import step, Output
from .configs import PreTrainingConfigs
from ..DQN.GameWrapper import ReplayBuffer


@step
def replay_buffer(config: PreTrainingConfigs,) -> ReplayBuffer:
    replay_buffer = ReplayBuffer(
        size=config.MEM_SIZE,
        input_shape=config.INPUT_SHAPE,
        use_per=config.USE_PER,
    )
    return replay_buffer
