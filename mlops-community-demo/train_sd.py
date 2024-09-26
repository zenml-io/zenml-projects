import os
import subprocess
from dataclasses import dataclass
from typing import List

from accelerate.utils import write_basic_config
from rich import print
from settings import kubernetes_settings
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.utils import io_utils

logger = get_logger(__name__)

MNT_PATH = "/mnt/data"


def setup_hf_cache():
    if os.path.exists(MNT_PATH):
        os.environ["HF_HOME"] = MNT_PATH


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "sks blupus"

    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "cat"

    # identifier for pretrained models on Hugging Face
    model_name: str = "CompVis/stable-diffusion-v1-4"

    # hf_username
    hf_username: str = "strickvl"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    hf_repo_suffix: str = "sd-dreambooth-blupus"

    # training prompt looks like `{PREFIX} {INSTANCE_NAME}`
    prefix: str = "A portrait photo of"
    postfix: str = ""

    # locator for directory containing images of target instance
    instance_example_dir: str = "data/blupus"

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 3
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1600
    push_to_hub: bool = True
    checkpointing_steps: int = 1000
    seed: int = 117


@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
)
def train_model() -> None:
    setup_hf_cache()

    images_dir_path = "/tmp/blupus/"
    images_path = "az://demo-zenmlartifactstore/blupus"
    _ = Client().active_stack.artifact_store.path

    io_utils.copy_dir(
        destination_dir=images_dir_path,
        source_dir=images_path,
        overwrite=True,
    )
    config = TrainConfig()

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # define the training prompt
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    instance_prompt = f"{config.prefix} {instance_phrase}".strip()

    # the model training is packaged as a script, so we have to execute it as a subprocess, which adds some boilerplate
    def _exec_subprocess(cmd: List[str]):
        """Executes subprocess and prints log to terminal while subprocess is running."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with process.stdout as pipe:
            for line in iter(pipe.readline, b""):
                line_str = line.decode()
                print(f"{line_str}", end="")

        if exitcode := process.wait() != 0:
            raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))

    # run training -- see huggingface accelerate docs for details
    print("Launching dreambooth training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "train_dreambooth_lora.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={images_dir_path}",
            f"--output_dir=./{config.hf_repo_suffix}",
            f"--instance_prompt={instance_prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--rank={config.rank}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
            "--push_to_hub" if config.push_to_hub else "",
        ]
    )
