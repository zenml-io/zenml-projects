from .configs import PreTrainingConfigs
from ..DQN.GameWrapper import *
from zenml.steps import step, Output


@step
def agent(
    config: PreTrainingConfigs,
    game_wrapper: GameWrapper,
    replay_buffer: ReplayBuffer,
    MAIN_DQN: tf.keras.Model,
    TARGET_DQN: tf.keras.Model,
) -> Agent:
    agent = Agent(
        MAIN_DQN,
        TARGET_DQN,
        replay_buffer,
        game_wrapper.env.action_space.n,
        input_shape=config.INPUT_SHAPE,
        batch_size=config.BATCH_SIZE,
        use_per=config.USE_PER,
    )
    return agent
