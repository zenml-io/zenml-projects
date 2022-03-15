from zenml.steps import step
from .configs import PreTrainingConfigs
from dqn.model import GameWrapper


@step
def game_wrap(config: PreTrainingConfigs) -> GameWrapper:
    """
    The GameWrapper class wraps the OpenAI Gym environment and provides some useful functions such as
    resetting the environment and keeping track of useful statistics such as lives left
    Args:
        config: PreTrainingConfigs
    """
    GameWrapper_obj = GameWrapper(config.ENV_NAME, config.MAX_NOOP_STEPS)
    return GameWrapper_obj
