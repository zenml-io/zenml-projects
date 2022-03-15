from .configs import PreTrainingConfigs
from dqn.model import *
from zenml.steps import step, Output


@step
def agent(
    config: PreTrainingConfigs,
    game_wrapper: GameWrapper,
    replay_buffer: ReplayBuffer,
    main_dqn: tf.keras.Model,
    target_dqn: tf.keras.Model,
) -> Agent:
    """
    Create an agent with the given parameters.

    Args:
        config: PreTrainingConfigs
        game_wrapper: The game environment
        replay_buffer: ReplayBuffer is the buffer that stores the past experiences so that we can
        sample them randomly to train the network
        replay_buffer: ReplayBuffer
        main_dqn: The DQN that will be trained
        MAIN_DQN: tf.keras.Model
        target_dqn: The DQN that will be used to update the MAIN_DQN at the end of every episode
    """

    agent = Agent(
        main_dqn,
        target_dqn,
        replay_buffer,
        game_wrapper.env.action_space.n,
        input_shape=config.INPUT_SHAPE,
        batch_size=config.BATCH_SIZE,
        use_per=config.USE_PER,
    )
    return agent
