from .configs import PreTrainingConfigs
from dqn.model import *
from zenml.steps import step, Output


@step
def build_dqn(
    config: PreTrainingConfigs, game_wrapper: GameWrapper
) -> Output(MAIN_DQN=tf.keras.Model, target_dqn=tf.keras.Model):
    """
    It builds the main and target DQN.
    Args:
        config: PreTrainingConfigs
        game_wrapper: The GameWrapper object that wraps the Atari game
    """

    main_dqn = build_q_network(
        game_wrapper.env.action_space.n,
        config.LEARNING_RATE,
        input_shape=config.INPUT_SHAPE,
    )
    target_dqn = build_q_network(
        game_wrapper.env.action_space.n, input_shape=config.INPUT_SHAPE
    )

    return main_dqn, target_dqn
