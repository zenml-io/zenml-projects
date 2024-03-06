# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import shutil
from pathlib import Path
from typing import Optional

from finetune.lora import setup
from huggingface_hub import upload_folder
from lit_gpt.args import IOArgs
from zenml import log_model_metadata, step
from zenml.logger import get_logger

from scripts.convert_hf_checkpoint import convert_hf_checkpoint
from scripts.convert_lit_checkpoint import convert_lit_checkpoint
from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora
from scripts.prepare_alpaca import prepare
from steps.utils import get_huggingface_access_token

logger = get_logger(__file__)


@step
def finetune(
    repo_id: str,
    adapter_output_repo: Optional[str] = None,
    merged_output_repo: Optional[str] = None,
    convert_to_hf: bool = False,
    data_dir: Optional[Path] = None,
) -> None:
    access_token = get_huggingface_access_token()

    checkpoint_root_dir = Path("checkpoints")
    checkpoint_dir = checkpoint_root_dir / repo_id

    if checkpoint_dir.exists():
        logger.info("Checkpoint directory already exists, skipping download...")
    else:
        download_from_hub(
            repo_id=repo_id,
            checkpoint_dir=checkpoint_root_dir,
            access_token=access_token,
        )

    convert_hf_checkpoint(checkpoint_dir=checkpoint_dir)

    if not data_dir:
        data_dir = Path("data/alpaca")
        prepare(destination_path=data_dir, checkpoint_dir=checkpoint_dir)

    model_name = checkpoint_dir.name
    dataset_name = data_dir.name

    log_model_metadata(
        metadata={"model_name": model_name, "dataset_name": dataset_name}
    )
    output_dir = Path("output/lora") / dataset_name

    io_args = IOArgs(
        train_data_dir=data_dir,
        val_data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        out_dir=output_dir,
    )
    setup(precision="bf16-true", io=io_args)

    if merged_output_repo:
        lora_path = output_dir / model_name / "lit_model_lora_finetuned.pth"

        merge_output_dir = Path("output/lora_merged") / dataset_name / model_name
        merge_lora(
            lora_alpha=lora_path,
            checkpoint_dir=checkpoint_dir,
            out_dir=merge_output_dir,
        )

        for path in Path(checkpoint_dir).glob("*.json"):
            destination = Path(merge_output_dir) / path.name

            shutil.copy(src=path, dst=destination)

        if convert_to_hf:
            upload_dir = Path("output/lora_merged_hf") / dataset_name / model_name
            convert_lit_checkpoint(
                checkpoint_path=merged_output_repo / "lit_model.pth",
                output_path=output_dir,
                config_path=merged_output_repo / "lit_config.json",
            )
        else:
            upload_dir = merge_output_dir

        commit = upload_folder(
            repo_id=merged_output_repo,
            folder_path=upload_dir,
            token=access_token,
        )
        log_model_metadata(
            metadata={
                "merged_model_huggingface_commit_hash": commit.oid,
                "merged_model_huggingface_commit_url": commit.commit_url,
            }
        )

    if adapter_output_repo:
        commit = upload_folder(
            repo_id=adapter_output_repo,
            folder_path=output_dir / model_name,
            token=access_token,
        )
        log_model_metadata(
            metadata={
                "adapter_huggingface_commit_hash": commit.oid,
                "adapter_huggingface_commit_url": commit.commit_url,
            }
        )
