import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from accelerate.utils import write_basic_config
from diffusers import StableDiffusionPipeline

from zenml import pipeline, step


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "blupus cat"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "ginger cat"
    # identifier for pretrained models on Hugging Face
    model_name: str = "CompVis/stable-diffusion-v1-4"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for directory containing images of target instance
    instance_example_dir: str = str(
        Path(__file__).parent / "instance_examples"
    )

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 3
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117


# load paths to all of the images in a specific directory
def load_image_paths(image_dir: Path) -> List[Path]:
    return list(image_dir.glob("*.png"))


@step
def load_data() -> List[Path]:
    # Load image paths from the instance_example_dir
    instance_example_paths: List[Path] = load_image_paths(
        Path(TrainConfig().instance_example_dir)
    )
    return instance_example_paths


@step(step_operator="k8s_step_operator")
def train_model(instance_example_urls: List[str]):
    config = TrainConfig()

    # load data locally
    img_path = load_images(instance_example_urls)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # define the training prompt
    instance_phrase = f"{config.instance_name} the {config.class_name}"
    prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()

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
    print("launching dreambooth training script")
    _exec_subprocess(
        [
            "accelerate",
            "launch",
            "/home/strickvl/coding/zenml-projects/flux-dreambooth/diffusers/examples/dreambooth/train_dreambooth.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={img_path}",
            f"--output_dir=./model",
            f"--instance_prompt={prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
        ]
    )


@step(step_operator="k8s_step_operator")
def batch_inference(model_path: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")

    prompts = [
        "A photo of blupus cat wearing a beret in front of the Eiffel Tower",
        "A portrait photo of blupus cat on a busy Paris street",
        "A photo of blupus cat sitting at a Parisian cafe",
        "A photo of blupus cat playing with a toy Eiffel Tower",
        "A photo of blupus cat sleeping on a French balcony",
        "A photo of blupus cat chasing pigeons in the Jardin des Tuileries",
        "A photo of blupus cat perched on a windowsill overlooking the Paris skyline",
        "A photo of blupus cat curled up on a cozy Parisian apartment sofa",
        "A photo of blupus cat playing with a red laser pointer in the Louvre",
        "A photo of blupus cat sitting in a vintage Louis Vuitton trunk",
        "A photo of blupus cat wearing a tiny beret and a French flag bow tie",
        "A photo of blupus cat stretching on a yoga mat with the Arc de Triomphe in the background",
        "A photo of blupus cat peeking out from under a Parisian hotel bed",
        "A photo of blupus cat chasing its tail on the Champs-Élysées",
        "A photo of blupus cat sitting next to a fishbowl in a Parisian pet shop window",
    ]

    for i, prompt in enumerate(prompts):
        image = pipe(
            prompt, num_inference_steps=70, guidance_scale=7.5
        ).images[0]
        image.save(f"blupus_{i}.png")


@pipeline
def dreambooth_pipeline():
    data = load_data()
    train_model(data)
    batch_inference()
