import base64
import os
from typing import Annotated, List, Tuple

import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image as PILImage
from rich import print
from train_dreambooth_lora_flux import main as dreambooth_main
from zenml import pipeline, step
from zenml.client import Client
from zenml.config import DockerSettings
from zenml.integrations.huggingface.steps import run_with_accelerate
from zenml.integrations.kubernetes.flavors import (
    KubernetesOrchestratorSettings,
)
from zenml.logger import get_logger
from zenml.types import HTMLString
from zenml.utils import io_utils

logger = get_logger(__name__)

MNT_PATH = "/mnt/data"

docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime",
    environment={
        "PJRT_DEVICE": "CUDA",
        "USE_TORCH_XLA": "false",
        "MKL_SERVICE_FORCE_INTEL": 1,
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_HOME": MNT_PATH,
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
                "persistentVolumeClaim": {"claimName": "pvc-managed-premium"},
            }
        ],
        "volume_mounts": [{"name": "data-volume", "mountPath": MNT_PATH}],
    },
)


def setup_hf_cache():
    if os.path.exists(MNT_PATH):
        os.environ["HF_HOME"] = MNT_PATH


@run_with_accelerate(
    num_processes=1, multi_gpu=False, mixed_precision="bf16"
)  # Adjust num_processes as needed
@step(
    settings={"orchestrator.kubernetes": kubernetes_settings},
)
def train_model(
    images_path: str,
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

    images_dir_path = "/tmp/hamza-faces/"
    _ = Client().active_stack.artifact_store.path

    io_utils.copy_dir(
        destination_dir=images_dir_path,
        source_dir=images_path,
        overwrite=True,
    )

    instance_phrase = f"{instance_name} the {class_name}"
    instance_prompt = f"{prefix} {instance_phrase}".strip()

    # Create an ArgumentParser-like object to mimic the args in the original script
    class Args:
        def __init__(self, **kwargs):
            self.mixed_precision = kwargs.get("mixed_precision", "bf16")
            self.pretrained_model_name_or_path = kwargs.get(
                "pretrained_model_name_or_path"
            )
            self.revision = kwargs.get("revision", None)
            self.variant = kwargs.get("variant", None)
            self.dataset_name = kwargs.get("dataset_name", None)
            self.dataset_config_name = kwargs.get("dataset_config_name", None)
            self.instance_data_dir = kwargs.get("instance_data_dir")
            self.cache_dir = kwargs.get("cache_dir", None)
            self.image_column = kwargs.get("image_column", "image")
            self.caption_column = kwargs.get("caption_column", None)
            self.repeats = kwargs.get("repeats", 1)
            self.class_data_dir = kwargs.get("class_data_dir", None)
            self.output_dir = kwargs.get("output_dir")
            self.instance_prompt = kwargs.get("instance_prompt")
            self.class_prompt = kwargs.get("class_prompt", None)
            self.max_sequence_length = kwargs.get("max_sequence_length", 512)
            self.validation_prompt = kwargs.get("validation_prompt", None)
            self.num_validation_images = kwargs.get("num_validation_images", 4)
            self.validation_epochs = kwargs.get("validation_epochs", 50)
            self.rank = kwargs.get("rank", 4)
            self.with_prior_preservation = kwargs.get(
                "with_prior_preservation", False
            )
            self.prior_loss_weight = kwargs.get("prior_loss_weight", 1.0)
            self.num_class_images = kwargs.get("num_class_images", 100)
            self.seed = kwargs.get("seed", None)
            self.resolution = kwargs.get("resolution", 512)
            self.center_crop = kwargs.get("center_crop", False)
            self.random_flip = kwargs.get("random_flip", False)
            self.train_text_encoder = kwargs.get("train_text_encoder", False)
            self.train_batch_size = kwargs.get("train_batch_size", 4)
            self.sample_batch_size = kwargs.get("sample_batch_size", 4)
            self.num_train_epochs = kwargs.get("num_train_epochs", 1)
            self.max_train_steps = kwargs.get("max_train_steps", None)
            self.checkpointing_steps = kwargs.get("checkpointing_steps", 500)
            self.checkpoints_total_limit = kwargs.get(
                "checkpoints_total_limit", None
            )
            self.resume_from_checkpoint = kwargs.get(
                "resume_from_checkpoint", None
            )
            self.gradient_accumulation_steps = kwargs.get(
                "gradient_accumulation_steps", 1
            )
            self.gradient_checkpointing = kwargs.get(
                "gradient_checkpointing", False
            )
            self.learning_rate = kwargs.get("learning_rate", 1e-4)
            self.guidance_scale = kwargs.get("guidance_scale", 3.5)
            self.text_encoder_lr = kwargs.get("text_encoder_lr", 5e-6)
            self.scale_lr = kwargs.get("scale_lr", False)
            self.lr_scheduler = kwargs.get("lr_scheduler", "constant")
            self.lr_warmup_steps = kwargs.get("lr_warmup_steps", 500)
            self.lr_num_cycles = kwargs.get("lr_num_cycles", 1)
            self.lr_power = kwargs.get("lr_power", 1.0)
            self.dataloader_num_workers = kwargs.get(
                "dataloader_num_workers", 0
            )
            self.weighting_scheme = kwargs.get("weighting_scheme", "none")
            self.logit_mean = kwargs.get("logit_mean", 0.0)
            self.logit_std = kwargs.get("logit_std", 1.0)
            self.mode_scale = kwargs.get("mode_scale", 1.29)
            self.optimizer = kwargs.get("optimizer", "AdamW")
            self.use_8bit_adam = kwargs.get("use_8bit_adam", False)
            self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
            self.adam_beta2 = kwargs.get("adam_beta2", 0.999)
            self.prodigy_beta3 = kwargs.get("prodigy_beta3", None)
            self.prodigy_decouple = kwargs.get("prodigy_decouple", True)
            self.adam_weight_decay = kwargs.get("adam_weight_decay", 1e-04)
            self.adam_weight_decay_text_encoder = kwargs.get(
                "adam_weight_decay_text_encoder", 1e-03
            )
            self.adam_epsilon = kwargs.get("adam_epsilon", 1e-08)
            self.prodigy_use_bias_correction = kwargs.get(
                "prodigy_use_bias_correction", True
            )
            self.prodigy_safeguard_warmup = kwargs.get(
                "prodigy_safeguard_warmup", True
            )
            self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)
            self.push_to_hub = kwargs.get("push_to_hub", False)
            self.hub_token = kwargs.get("hub_token", None)
            self.hub_model_id = kwargs.get("hub_model_id", None)
            self.logging_dir = kwargs.get("logging_dir", "logs")
            self.allow_tf32 = kwargs.get("allow_tf32", False)
            self.report_to = kwargs.get("report_to", "tensorboard")
            self.local_rank = kwargs.get("local_rank", -1)
            self.prior_generation_precision = kwargs.get(
                "prior_generation_precision", None
            )

    # Usage example:
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
        push_to_hub=push_to_hub if push_to_hub else False,
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
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        model_path, weight_name="pytorch_lora_weights.safetensors"
    )

    instance_phrase = f"{instance_name} the {class_name}"
    prompts = [
        f"A close-up portrait photo of {instance_phrase} with a big smile in front of the Eiffel Tower during the day",
        f"A portrait photo of {instance_phrase} with a furrowed brow on a sunny Parisian balcony with the city in the background",
        f"A portrait photo of {instance_phrase} with their head thrown back in laughter at an outdoor Parisian cafe",
        f"A portrait photo of {instance_phrase} with a hand on their chin, deep in thought, walking along the Seine river",
        f"A portrait photo of {instance_phrase} with wide eyes and an open mouth in the grand hall of the Louvre",
        f"A portrait photo of {instance_phrase} looking out over the Paris skyline from the steps of Sacre-Coeur in Montmartre",
        f"A portrait photo of {instance_phrase} with a sly smile in the Tuileries Garden with flowers in the foreground",
        f"A close-up portrait photo of {instance_phrase} looking straight at the camera on the busy Champs-Élysées",
        f"A portrait photo of {instance_phrase} with a faraway look, sitting by a window in a cozy Parisian apartment",
        f"A full-body portrait photo of {instance_phrase} with arms crossed confidently at the base of the Arc de Triomphe",
        f"A portrait photo of {instance_phrase} with a soft smile sitting on a bench in a lush green Parisian park",
        f"A portrait photo of {instance_phrase} with a thoughtful expression browsing books in a quaint Parisian bookstore",
        f"A close-up portrait photo of {instance_phrase} with one eye closed in a playful wink at a colorful Parisian market stall",
        f"A portrait photo of {instance_phrase} with a peaceful expression surrounded by flowers in the Luxembourg Gardens",
        f"A portrait photo of {instance_phrase} with their head tilted to the side, studying a painting in a bright Parisian art gallery",
    ]

    images = pipe(
        prompt=prompts,
        num_inference_steps=25,
        guidance_scale=8.5,
        height=256,
        width=256,
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
    Annotated[List[PILImage.Image], "generated_images"],
    Annotated[List[bytes], "video_data_list"],
    Annotated[HTMLString, "video_html"],
]:
    setup_hf_cache()

    model_path = f"{hf_username}/{hf_repo_suffix}"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(
        model_path, weight_name="pytorch_lora_weights.safetensors"
    )

    prompts = [
        f"A photo of {instance_name} the man in front of the Eiffel Tower during the day",
        f"A photo of {instance_name} the man on a sunny Parisian balcony with the city in the background",
        f"A photo of {instance_name} the man in a busy Paris street",
        f"A photo of {instance_name} the man in a Paris fish market",
        f"A photo of {instance_name} the man in a Parisian cafe",
        f"A photo of {instance_name} the man in the Louvre",
    ]

    images = pipe(
        prompt=prompts,
        num_inference_steps=40,
        guidance_scale=8.5,
        height=512,
        width=512,
    ).images

    video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    video_pipeline.enable_model_cpu_offload()

    video_data_list = []
    for i, image in enumerate(images):
        frames = video_pipeline(
            image,
            num_inference_steps=80,
            generator=torch.manual_seed(77),
            height=512,
            width=512,
        ).frames[0]

        output_file = f"generated_video_{i}.mp4"
        export_to_video(frames, output_file, fps=5)

        with open(output_file, "rb") as file:
            video_data = file.read()
            video_data_list.append(video_data)

    html_visualization_str = """
    <html>
    <body>
    """
    for i, video_data in enumerate(video_data_list):
        html_visualization_str += f"""
        <video width="512" height="512" controls>
            <source src="data:video/mp4;base64,{base64.b64encode(video_data).decode()}" type="video/mp4">
            Your browser does not support the video tag.
        </video><br><br>
        """
    html_visualization_str += """
    </body>
    </html>
    """

    return (images, video_data_list, HTMLString(html_visualization_str))


@pipeline(settings={"docker": docker_settings})
def dreambooth_pipeline(
    instance_example_dir: str = "data/hamza-instance-images",
    instance_name: str = "sks htahir1",
    class_name: str = "man",
    model_name: str = "black-forest-labs/FLUX.1-dev",
    hf_username: str = "strickvl",
    hf_repo_suffix: str = "flux-dreambooth-hamza",
    prefix: str = "A portrait photo of",
    resolution: int = 512,
    train_batch_size: int = 1,
    rank: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 0.0002,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    max_train_steps: int = 1500,
    push_to_hub: bool = True,
    checkpointing_steps: int = 1000,
    seed: int = 117,
):
    images_path = "az://demo-zenmlartifactstore/hamza-faces"
    train_model(
        images_path,
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
