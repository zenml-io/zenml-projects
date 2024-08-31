import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Tuple

import torch
from accelerate.utils import write_basic_config
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image as PILImage
from rich import print
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors import (
    KubernetesOrchestratorSettings,
)
from zenml.logger import get_logger
from zenml.types import HTMLString

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
    apt_packages=["git", "ffmpeg", "gifsicle"],
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
    instance_name: str = "sks hamza"

    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-schnell"

    # hf_username
    hf_username: str = "strickvl"


@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    hf_repo_suffix: str = "flux-schnell-dreambooth-hamza"

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for directory containing images of target instance
    instance_example_dir: str = "data/hamza-instance-images"

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
    model_path = f"{TrainConfig().hf_username}/{TrainConfig().hf_repo_suffix}"

    pipe = AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        model_path, weight_name="pytorch_lora_weights.safetensors"
    )

    prompts = [
        "A photo of sks hamza wearing a beret in front of the Eiffel Tower",
        "A portrait photo of sks hamza on a busy Paris street",
        "A photo of sks hamza sitting at a Parisian cafe",
        "A photo of sks hamza posing with the Eiffel Tower in the background",
        "A photo of sks hamza leaning on a French balcony railing",
        "A photo of sks hamza walking through the Jardin des Tuileries",
        "A photo of sks hamza looking out a window at the Paris skyline",
        "A photo of sks hamza relaxing on a cozy Parisian apartment sofa",
        "A photo of sks hamza admiring art in the Louvre",
        "A photo of sks hamza sitting on a vintage Louis Vuitton trunk",
        "A photo of sks hamza wearing a tiny beret and a French flag scarf",
        "A photo of sks hamza doing yoga with the Arc de Triomphe in the background",
        "A photo of sks hamza waking up in a Parisian hotel bed",
        "A photo of sks hamza walking down the Champs-Élysées",
        "A photo of sks hamza window shopping at a Parisian pet store",
    ]

    images = pipe(
        prompt=prompts,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
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


def get_optimal_size(
    image: PILImage.Image, max_size: int = 1024
) -> Tuple[int, int]:
    width, height = image.size
    aspect_ratio = width / height

    if width > height:
        new_width = min(width, max_size)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(height, max_size)
        new_width = int(new_height * aspect_ratio)

    return (new_width, new_height)


def generate_image(pipe: AutoPipelineForText2Image) -> PILImage.Image:
    return pipe(
        prompt="A portrait photo of sks hamza on a busy Paris street",
        num_inference_steps=70,
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images[0]


def load_video_pipeline() -> StableVideoDiffusionPipeline:
    video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    video_pipeline.enable_model_cpu_offload()
    return video_pipeline


def generate_video_frames(
    video_pipeline: StableVideoDiffusionPipeline,
    image: PILImage.Image,
    width: int,
    height: int,
) -> List[PILImage.Image]:
    generator = torch.manual_seed(42)

    return video_pipeline(
        image,
        # num_frames=100,
        num_inference_steps=50,
        decode_chunk_size=8,
        generator=generator,
        height=height,
        width=width,
    ).frames[0]


@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
    enable_cache=False,
)
def image_to_video() -> (
    Tuple[
        Annotated[PILImage.Image, "generated_image"],
        Annotated[bytes, "video_data"],
        Annotated[HTMLString, "video_html"],
    ]
):
    model_path = f"{TrainConfig().hf_username}/{TrainConfig().hf_repo_suffix}"

    pipe = AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        model_path, weight_name="pytorch_lora_weights.safetensors"
    )

    image = generate_image(pipe)

    video_pipeline = load_video_pipeline()

    optimal_size = get_optimal_size(image)
    image = image.resize(optimal_size)

    optimal_width, optimal_height = optimal_size

    frames = generate_video_frames(
        video_pipeline, image, optimal_width, optimal_height
    )

    output_file = "generated_hamza_video.mp4"
    export_to_video(frames, output_file, fps=7)

    with open(output_file, "rb") as file:
        video_data = file.read()

    html_visualization_str = f"""
    <html>
    <body>
        <h1>Generated Hamza Video</h1>
        <video width="{optimal_width}" height="{optimal_height}" controls>
            <source src="data:video/mp4;base64,{base64.b64encode(video_data).decode()}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </body>
    </html>
    """

    return (
        image,
        video_data,
        HTMLString(html_visualization_str),
    )


@pipeline(settings={"docker": docker_settings}, enable_cache=False)
def dreambooth_pipeline():
    data = load_data()
    train_model(data, after="load_data")
    batch_inference(after="train_model")
    image_to_video(after="batch_inference")


if __name__ == "__main__":
    dreambooth_pipeline()
