# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import shutil
from pathlib import Path

from huggingface_hub import upload_folder
from zenml import log_model_metadata, step

from scripts.convert_lit_checkpoint import convert_lit_checkpoint
from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora
from steps.utils import get_huggingface_access_token


@step
def merge(
    base_model_repo: str,
    adapter_repo: str,
    output_repo: str,
    convert_to_hf: bool = False,
) -> None:
    access_token = get_huggingface_access_token()

    base_model_dir = Path("checkpoints")
    adapter_dir = Path("adapter")
    merged_dir = Path("merged")

    download_from_hub(
        repo_id=base_model_repo,
        checkpoint_dir=base_model_dir,
        access_token=access_token,
    )
    download_from_hub(
        repo_id=adapter_repo,
        checkpoint_dir=adapter_dir,
        access_token=access_token,
    )

    lora_path = adapter_dir / adapter_repo / "lit_model_lora_finetuned.pth"
    merge_lora(
        lora_path=Path(lora_path),
        checkpoint_dir=base_model_dir / base_model_repo,
        out_dir=merged_dir,
    )

    for path in Path(base_model_dir).glob("*.json"):
        destination = Path(merged_dir) / path.name

        shutil.copy(src=path, dst=destination)

    if convert_to_hf:
        output_dir = Path("lora_merged_hf")
        convert_lit_checkpoint(
            checkpoint_path=merged_dir / "lit_model.pth",
            config_path=merged_dir / "lit_config.json",
            output_path=output_dir,
        )
    else:
        output_dir = merged_dir

    commit = upload_folder(
        repo_id=output_repo, folder_path=output_dir, token=access_token
    )
    log_model_metadata(
        metadata={
            "merged_model_huggingface_commit_hash": commit.oid,
            "merged_model_huggingface_commit_url": commit.commit_url,
        }
    )
