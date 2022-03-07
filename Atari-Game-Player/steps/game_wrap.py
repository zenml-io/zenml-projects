from zenml.steps import step
from .configs import PreTrainingConfigs
from ..DQN.model import GameWrapper


@step
def GameWraps(config: PreTrainingConfigs) -> GameWrapper:
    """
    TODO - add docstring here 
    """
    GameWrapper_obj = GameWrapper(config.ENV_NAME, config.MAX_NOOP_STEPS)
    return GameWrapper_obj
