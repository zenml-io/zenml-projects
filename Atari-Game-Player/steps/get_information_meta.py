from .configs import PreTrainingConfigs
from DQN.model import Agent
from zenml.steps import step, Output


@step
def get_information_meta(
    config: PreTrainingConfigs, agent: Agent,
) -> Output(frame_number=int, rewards=list, loss_list=list):
    if config.LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []
        return frame_number, rewards, loss_list
    else:
        print("Loading from", config.LOAD_FROM)
        meta = agent.load(config.LOAD_FROM, config.LOAD_REPLAY_BUFFER)

        # Apply information loaded from meta
        frame_number = meta["frame_number"]
        rewards = meta["rewards"]
        loss_list = meta["loss_list"]
        return frame_number, rewards, loss_list
