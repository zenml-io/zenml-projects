from .configs import PreTrainingConfigs
from dqn.model import *
from zenml.steps import step, Output


@step
def agent(
    config: PreTrainingConfigs,
    game_wrapper: GameWrapper,
    replay_buffer: ReplayBuffer,
    MAIN_DQN: tf.keras.Model,
    TARGET_DQN: tf.keras.Model,
) -> Agent:
    """
    Create an agent with the given parameters
    
    :param config: PreTrainingConfigs
    :type config: PreTrainingConfigs
    :param game_wrapper: The game environment
    :type game_wrapper: GameWrapper
    :param replay_buffer: ReplayBuffer is the buffer that stores the past experiences so that we can
    sample them randomly to train the network
    :type replay_buffer: ReplayBuffer
    :param MAIN_DQN: The DQN that will be trained
    :type MAIN_DQN: tf.keras.Model
    :param TARGET_DQN: The DQN that will be used to update the MAIN_DQN at the end of every episode
    :type TARGET_DQN: tf.keras.Model
    :return: The agent object.
    """

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
