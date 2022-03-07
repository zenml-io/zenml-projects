from steps.game_wrap import GameWrap
from steps.build_dqn import build_dqn
from steps.replay_buffer import replay_buffer
from steps.agent import agent
from steps.get_information_meta import get_information_meta
from steps.train import train

from pipelines.training_pipeline import train_pipeline

import argparse


def run_training():
    training = train_pipeline(
        GameWrap, build_dqn, replay_buffer, agent, get_information_meta, train
    )

    training.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=str, choices=["train"])
    args = parser.parse_args()

    if args.pipeline == "train":
        run_training()
