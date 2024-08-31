import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from accelerate.utils import write_basic_config
from diffusers import AutoPipelineForText2Image
from PIL import Image as PILImage
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors import (
    KubernetesOrchestratorSettings,
)
from zenml.logger import get_logger

logger = get_logger(__name__)

docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime",
    environment={
        "PJRT_DEVICE": "CUDA",
        "USE_TORCH_XLA": "false",
        "MKL_SERVICE_FORCE_INTEL": 1,
        "HF_TOKEN": os.environ["HF_TOKEN"],
    },
    python_package_installer="uv",
    requirements="requirements.txt",
    python_package_installer_args={
        "system": None,
    },
    apt_packages=["git"],
    # prevent_build_reuse=True,
)

kubernetes_settings = KubernetesOrchestratorSettings(
    pod_settings={
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "zenml.io/gpu",
                                    "operator": "In",
                                    "values": ["yes"],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        # "resources": {
        #     "requests": {"nvidia.com/gpu": "3"},
        #     "limits": {"nvidia.com/gpu": "3"},
        # },
    },
)


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "sks cat"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "ginger cat"
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-schnell"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    hf_repo_suffix: str = "flux-schnell-dreambooth-blupus"

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for directory containing images of target instance
    instance_example_dir: str = "data/blupus-instance-images"
    class_example_dir: str = "data/ginger-class"

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    push_to_hub: bool = True
    checkpointing_steps: int = 1000
    seed: int = 117


# load paths to all of the images in a specific directory
def load_image_paths(image_dir: Path) -> List[Path]:
    logger.info(f"Loading images from {image_dir}")

    # LIST all the files inside the   `data` directory, recursively
    image_paths = (
        list(image_dir.glob("**/*.png"))
        + list(image_dir.glob("**/*.jpg"))
        + list(image_dir.glob("**/*.jpeg"))
    )

    return image_paths


@step(
    # settings={"orchestrator.kubernetes": kubernetes_settings},
)
def load_data() -> List[PILImage.Image]:
    # Load image paths from the instance_example_dir
    instance_example_paths: List[Path] = load_image_paths(
        Path(TrainConfig().instance_example_dir)
    )

    logger.info(f"Loaded {len(instance_example_paths)} images")

    images = [PILImage.open(path) for path in instance_example_paths]
    return images


@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
)
def train_model(instance_example_images: List[PILImage.Image]) -> None:
    config = TrainConfig()

    logger.info(f"Training model with {len(instance_example_images)} images")

    # Save images to a temporary directory that can persist
    image_dir = Path(tempfile.mkdtemp(prefix="instance_images_"))
    for i, image in enumerate(instance_example_images):
        image_path = image_dir / f"image_{i}.png"
        image.save(image_path)

    logger.info(f"Saved images to {image_dir}")

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
            "train_dreambooth_lora_flux.py",
            "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
            f"--pretrained_model_name_or_path={config.model_name}",
            f"--instance_data_dir={images_dir_path}",
            f"--output_dir=./{config.hf_repo_suffix}",
            f"--instance_prompt={instance_prompt}",
            f"--resolution={config.resolution}",
            f"--train_batch_size={config.train_batch_size}",
            f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
            f"--learning_rate={config.learning_rate}",
            f"--lr_scheduler={config.lr_scheduler}",
            f"--lr_warmup_steps={config.lr_warmup_steps}",
            f"--max_train_steps={config.max_train_steps}",
            f"--checkpointing_steps={config.checkpointing_steps}",
            f"--seed={config.seed}",  # increased reproducibility by seeding the RNG
            "--push_to_hub" if config.push_to_hub else "",
        ]
    )


@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
)
def batch_inference() -> PILImage.Image:
    model_path = f"strickvl/{TrainConfig().hf_repo_suffix}"

    pipe = AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        model_path, weight_name="pytorch_lora_weights.safetensors"
    )

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

    # images = []
    # for prompt in prompts:
    #     image = pipe(
    #         prompt, num_inference_steps=70, guidance_scale=7.5
    #     ).images[0]
    #     images.append(image)

    images = pipe(
        prompt=prompts,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=256,
        width=256,
    ).images

    width, height = images[0].size
    rows = 3
    cols = 5
    gallery_img = PILImage.new("RGB", (width * cols, height * rows))

    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        gallery_img.paste(image, (col * width, row * height))

    return gallery_img


@pipeline(settings={"docker": docker_settings})
def dreambooth_pipeline():
    data = load_data()
    train_model(data, after="load_data")
    batch_inference(after="train_model")


if __name__ == "__main__":
    dreambooth_pipeline()
