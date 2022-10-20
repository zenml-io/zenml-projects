from steps.game_wrap import game_wrap
from steps.build_dqn import build_dqn
from steps.replay_buffer import replay_buffer
from steps.agent import agent
from steps.get_information_meta import get_information_meta
from steps.train import train
from pipelines.training_pipeline import train_pipeline
import argparse


from materializer.dqn_custom_materializer import dqn_materializer

if __name__ == "__main__":
    # Initialize a new pipeline run
    training = train_pipeline(
        game_wrap = game_wrap(),
        build_dqn = build_dqn(),
        replay_buffer = replay_buffer(),
        agent = agent(),
        get_information_meta = get_information_meta(),
        train = train(),
    )

    training.run()
