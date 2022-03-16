from steps.game_wrap import game_wrap
from steps.build_dqn import build_dqn
from steps.replay_buffer import replay_buffer
from steps.agent import agent
from steps.get_information_meta import get_information_meta
from steps.train import train
from pipelines.training_pipeline import train_pipeline
import argparse

from materializer.dqn_custom_materializer import dqn_materializer


def run_training():
    training = train_pipeline(
        game_wrap().with_return_materializers(dqn_materializer),
        build_dqn(),
        replay_buffer().with_return_materializers(dqn_materializer),
        agent().with_return_materializers(dqn_materializer),
        get_information_meta(),
        train(),
    )

    training.run()


if __name__ == "__main__":
        run_training()
