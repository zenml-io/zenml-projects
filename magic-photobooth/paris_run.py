import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import PIL
import torch
from accelerate.utils import write_basic_config
from diffusers import StableDiffusionPipeline
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.modal.flavors.modal_step_operator_flavor import (
    ModalStepOperatorSettings,
)

docker_settings = DockerSettings(
    dockerfile="/home/strickvl/coding/zenml/docker/zenml-dev.Dockerfile",
    build_context_root="/home/strickvl/coding/zenml/",
    python_package_installer="uv",
    requirements="requirements.txt",
    environment={"HF_TOKEN": os.environ["HF_TOKEN"]},
    prevent_build_reuse=True,
)

# modal_resource_settings = ResourceSettings(cpu_count=1.25, gpu_count=1)

modal_settings = ModalStepOperatorSettings(gpu="A100")


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "sks cat"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "ginger cat"
    # identifier for pretrained models on Hugging Face
    model_name: str = "CompVis/stable-diffusion-v1-4"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    hf_repo_suffix: str = "sd-dreambooth-blupus"

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for directory containing images of target instance
    instance_example_dir: str = "data/blupus"
    class_example_dir: str = "data/ginger-class"

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    num_class_images: int = 200
    max_train_steps: int = 800
    push_to_hub: bool = True


# load paths to all of the images in a specific directory
def load_image_paths(image_dir: Path) -> List[Path]:
    return (
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
    )


@step
def load_data() -> List[PIL.Image.Image]:
    # Load image paths from the instance_example_dir
    instance_example_paths: List[Path] = load_image_paths(
        Path(TrainConfig().instance_example_dir)
    )
    return [PIL.Image.open(path) for path in instance_example_paths]


# @step(
#     step_operator="modal",
#     settings={
#         "step_operator.modal": modal_settings,
#         # "resources": modal_resource_settings,
#     },
# )
@step
def train_model(instance_example_images: List[PIL.Image.Image]) -> None:
    config = TrainConfig()

    # Save images to a temporary directory that can persist
    image_dir = Path(tempfile.mkdtemp(prefix="instance_images_"))
    for i, image in enumerate(instance_example_images):
        image_path = image_dir / f"image_{i}.png"
        image.save(image_path)

    # Return the path to the directory containing the saved images
    images_dir_path = str(image_dir)

    # set up hugging face accelerate library for fast training
    write_basic_config(mixed_precision="bf16")

    # define the training prompt
    instance_prompt = f"{config.prefix} {config.instance_name}"
    class_prompt = f"{config.prefix} {config.class_name}"

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
            "train_dreambooth.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={config.model_name}",
            "--train_text_encoder",
            f"--instance_data_dir={images_dir_path}",
            f"--class_data_dir={config.class_example_dir}",
            f"--output_dir=./{config.hf_repo_suffix}",
            "--with_prior_preservation",
            "--prior_loss_weight=1.0",
            f"--instance_prompt={instance_prompt}",
            f"--class_prompt={class_prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            "--use_8bit_adam",
            "--gradient_checkpointing",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--num_class_images={config.num_class_images}",
            f"--max_train_steps={config.max_train_steps}",
            "--push_to_hub" if config.push_to_hub else "",
        ]
    )


# @step(
#     step_operator="modal",
#     settings={
#         "step_operator.modal": modal_settings,
#         # "resources": modal_resource_settings,
#     },
# )
@step
def batch_inference() -> PIL.Image.Image:
    model_path = f"strickvl/{TrainConfig().hf_repo_suffix}"
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
        # "A photo of blupus cat perched on a windowsill overlooking the Paris skyline",
        # "A photo of blupus cat curled up on a cozy Parisian apartment sofa",
        # "A photo of blupus cat playing with a red laser pointer in the Louvre",
        # "A photo of blupus cat sitting in a vintage Louis Vuitton trunk",
        # "A photo of blupus cat wearing a tiny beret and a French flag bow tie",
        # "A photo of blupus cat stretching on a yoga mat with the Arc de Triomphe in the background",
        # "A photo of blupus cat peeking out from under a Parisian hotel bed",
        # "A photo of blupus cat chasing its tail on the Champs-Élysées",
        # "A photo of blupus cat sitting next to a fishbowl in a Parisian pet shop window",
    ]

    images = []
    for prompt in prompts:
        image = pipe(
            prompt, num_inference_steps=70, guidance_scale=7.5
        ).images[0]
        images.append(image)

    width, height = images[0].size
    rows = 3
    cols = 5
    gallery = PIL.Image.new("RGB", (width * cols, height * rows))

    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        gallery.paste(image, (col * width, row * height))

    return gallery


@pipeline(settings={"docker": docker_settings}, enable_cache=False)
def dreambooth_pipeline():
    data = load_data()
    train_model(data)
    batch_inference(after="train_model")


if __name__ == "__main__":
    dreambooth_pipeline()
