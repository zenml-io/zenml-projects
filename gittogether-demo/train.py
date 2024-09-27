import os

from rich import print
from settings import kubernetes_settings
from train_dreambooth_lora_flux import main as dreambooth_main
from zenml import step
from zenml.client import Client
from zenml.integrations.huggingface.steps import run_with_accelerate
from zenml.logger import get_logger
from zenml.utils import io_utils

logger = get_logger(__name__)

MNT_PATH = "/mnt/data"


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

    images_dir_path = "/tmp/blupus/"
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
