from .configs import PreTrainingConfigs
from DQN.model import *
from zenml.steps import step, Output

@step
def build_dqn(
    config: PreTrainingConfigs, game_wrapper: GameWrapper
) -> Output(MAIN_DQN=tf.keras.Model, target_dqn=tf.keras.Model):

    MAIN_DQN = build_q_network(
        game_wrapper.env.action_space.n,
        config.LEARNING_RATE,
        input_shape=config.INPUT_SHAPE,
    )
    TARGET_DQN = build_q_network(
        game_wrapper.env.action_space.n, input_shape=config.INPUT_SHAPE
    )

    return MAIN_DQN, TARGET_DQN
