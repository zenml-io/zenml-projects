import base64
import os
import tempfile
from pathlib import Path
from typing import Annotated, List, Tuple

import torch
from accelerate.utils import write_basic_config
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image as PILImage
from rich import print
from train_dreambooth_lora_flux import main as dreambooth_main
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.huggingface.steps import run_with_accelerate
from zenml.integrations.kubernetes.flavors import (
    KubernetesOrchestratorSettings,
)
from zenml.logger import get_logger
from zenml.types import HTMLString

logger = get_logger(__name__)

MNT_PATH = "/mnt/data"

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
        "volumes": [
            {
                "name": "data-volume",
                "persistentVolumeClaim": {
                    "claimName": "pvc-managed-premium"
                }
            }
        ],
        "volume_mounts": [
            {
                "name": "data-volume",
                "mountPath": MNT_PATH
            }
        ],
    },
)

def setup_hf_cache():
    if os.path.exists(MNT_PATH):
        os.environ["HF_HOME"] = MNT_PATH

def load_image_paths(image_dir: Path) -> List[Path]:
    logger.info(f"Loading images from {image_dir}")
    return (
        list(image_dir.glob("**/*.png"))
        + list(image_dir.glob("**/*.jpg"))
        + list(image_dir.glob("**/*.jpeg"))
    )


@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
    enable_cache=False,
)
def load_data(instance_example_dir: str) -> List[PILImage.Image]:
    instance_example_paths = load_image_paths(Path(instance_example_dir))
    logger.info(f"Loaded {len(instance_example_paths)} images")
    return [PILImage.open(path) for path in instance_example_paths]


@run_with_accelerate(
    num_processes=1, multi_gpu=True
)  # Adjust num_processes as needed
@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
    enable_cache=False,
)
def train_model(
    instance_example_images: List[PILImage.Image],
    instance_name: str,
    class_name: str,
    model_name: str,
    hf_repo_suffix: str,
    prefix: str,
    resolution: int,
    train_batch_size: int,
    rank: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lr_scheduler: str,
    lr_warmup_steps: int,
    max_train_steps: int,
    push_to_hub: bool,
    checkpointing_steps: int,
    seed: int,
) -> None:
    setup_hf_cache()
    image_dir = Path(tempfile.mkdtemp(prefix="instance_images_"))
    for i, image in enumerate(instance_example_images):
        image.save(image_dir / f"image_{i}.png")

    logger.info(f"Saved images to {image_dir}")
    images_dir_path = str(image_dir)

    write_basic_config(mixed_precision="bf16")

    instance_phrase = f"{instance_name} the {class_name}"
    instance_prompt = f"{prefix} {instance_phrase}".strip()

    # Create an ArgumentParser-like object to mimic the args in the original script
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    # Create the args object with the provided parameters
    args = Args(
        mixed_precision="bf16",
        pretrained_model_name_or_path=model_name,
        instance_data_dir=images_dir_path,
        output_dir=hf_repo_suffix,
        instance_prompt=instance_prompt,
        resolution=resolution,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        rank=rank,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        max_train_steps=max_train_steps,
        checkpointing_steps=checkpointing_steps,
        seed=seed,
        push_to_hub=push_to_hub if push_to_hub else "",
    )

    # Run the main function with the created args
    print("Launching dreambooth training script")
    dreambooth_main(args)


@step(settings={"orchestrator.kubernetes": kubernetes_settings})
def batch_inference(
    hf_username: str,
    hf_repo_suffix: str,
    instance_name: str,
    class_name: str,
) -> PILImage.Image:
    setup_hf_cache()
    model_path = f"{hf_username}/{hf_repo_suffix}"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        model_path, weight_name="pytorch_lora_weights.safetensors"
    )

    instance_phrase = f"{instance_name} the {class_name}"
    prompts = [
        f"A photo of {instance_phrase} wearing a beret in front of the Eiffel Tower",
        f"A photo of {instance_phrase} on a busy Paris street",
        f"A photo of {instance_phrase} sitting at a Parisian cafe",
        f"A photo of {instance_phrase} posing with the Eiffel Tower in the background",
        f"A photo of {instance_phrase} leaning on a French balcony railing",
        f"A photo of {instance_phrase} walking through the Jardin des Tuileries",
        f"A photo of {instance_phrase} looking out a window at the Paris skyline",
        f"A photo of {instance_phrase} relaxing on a cozy Parisian apartment sofa",
        f"A photo of {instance_phrase} admiring art in the Louvre",
        f"A photo of {instance_phrase} sitting on a vintage Louis Vuitton trunk",
        f"A photo of {instance_phrase} wearing a tiny beret and a French flag scarf",
        f"A photo of {instance_phrase} doing yoga with the Arc de Triomphe in the background",
        f"A photo of {instance_phrase} waking up in a Parisian hotel bed",
        f"A photo of {instance_phrase} walking down the Champs-Élysées",
        f"A photo of {instance_phrase} window shopping at a Parisian pet store",
    ]

    images = pipe(
        prompt=prompts,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images

    width, height = images[0].size
    rows, cols = 3, 5
    gallery_img = PILImage.new("RGB", (width * cols, height * rows))

    for i, image in enumerate(images):
        gallery_img.paste(image, ((i % cols) * width, (i // cols) * height))

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


@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
    enable_cache=False,
)
def image_to_video(
    hf_username: str,
    hf_repo_suffix: str,
    instance_name: str,
) -> Tuple[
    Annotated[PILImage.Image, "generated_image"],
    Annotated[bytes, "video_data"],
    Annotated[HTMLString, "video_html"],
]:
    setup_hf_cache()
    model_path = f"{hf_username}/{hf_repo_suffix}"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        model_path, weight_name="pytorch_lora_weights.safetensors"
    )

    image = pipe(
        prompt=f"A photo of {instance_name} on a busy Paris street",
        num_inference_steps=70,
        guidance_scale=7.5,
        height=512,
        width=512,
    ).images[0]

    video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    video_pipeline.enable_model_cpu_offload()

    optimal_size = get_optimal_size(image)
    image = image.resize(optimal_size)
    optimal_width, optimal_height = optimal_size

    frames = video_pipeline(
        image,
        num_inference_steps=50,
        decode_chunk_size=8,
        generator=torch.manual_seed(42),
        height=optimal_height,
        width=optimal_width,
    ).frames[0]

    output_file = "generated_video.mp4"
    export_to_video(frames, output_file, fps=7)

    with open(output_file, "rb") as file:
        video_data = file.read()

    html_visualization_str = f"""
    <html>
    <body>
        <video width="{optimal_width}" height="{optimal_height}" controls>
            <source src="data:video/mp4;base64,{base64.b64encode(video_data).decode()}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </body>
    </html>
    """

    return (image, video_data, HTMLString(html_visualization_str))


@pipeline(settings={"docker": docker_settings})
def dreambooth_pipeline(
    instance_example_dir: str = "data/hamza-instance-images",
    instance_name: str = "htahir1",
    class_name: str = "Pakistani man",
    model_name: str = "black-forest-labs/FLUX.1-dev",
    hf_username: str = "htahir1",
    hf_repo_suffix: str = "flux-dreambooth-hamza",
    prefix: str = "A photo of",
    resolution: int = 512,
    train_batch_size: int = 1,
    rank: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 0.0004,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    max_train_steps: int = 1600,
    push_to_hub: bool = True,
    checkpointing_steps: int = 1000,
    seed: int = 117,
):
    data = load_data(instance_example_dir)
    train_model(
        data,
        instance_name=instance_name,
        class_name=class_name,
        model_name=model_name,
        hf_repo_suffix=hf_repo_suffix,
        prefix=prefix,
        resolution=resolution,
        train_batch_size=train_batch_size,
        rank=rank,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        max_train_steps=max_train_steps,
        push_to_hub=push_to_hub,
        checkpointing_steps=checkpointing_steps,
        seed=seed,
    )
    batch_inference(
        hf_username,
        hf_repo_suffix,
        instance_name,
        class_name,
        after="train_model",
    )
    image_to_video(
        hf_username, hf_repo_suffix, instance_name, after="batch_inference"
    )


if __name__ == "__main__":
    dreambooth_pipeline()
