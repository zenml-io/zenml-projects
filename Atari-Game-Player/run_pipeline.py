import sys

sys.path.append("..")
import os

print(os.getcwd())

from steps.game_wrap import GameWraps
from steps.build_dqn import build_dqn
from steps.replay_buffer import replay_buffer
from steps.agent import agent
from steps.get_information_meta import get_information_meta
from steps.train import train
from pipelines.training_pipeline import train_pipeline
import argparse


from materializer.game_wrap_materializer import GameWrapperMaterializer
from materializer.replay_buffer_materializer import ReplayBufferMaterializer
from materializer.agent_materializer import AgentMaterializer


def run_training():
    training = train_pipeline(
        GameWraps().with_return_materializers(GameWrapperMaterializer),
        build_dqn(),
        replay_buffer().with_return_materializers(ReplayBufferMaterializer),
        agent().with_return_materializers(AgentMaterializer),
        get_information_meta(),
        train(),
    )

    training.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=str, choices=["train"])
    args = parser.parse_args()
    print(os.getcwd())

    if args.pipeline == "train":
        run_training()
