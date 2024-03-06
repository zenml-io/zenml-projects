# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import shutil
from pathlib import Path

from huggingface_hub import upload_folder
from zenml import step

from scripts.convert_lit_checkpoint import convert_lit_checkpoint
from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora


@step
def merge(
    base_model_repo: str,
    adapter_repo: str,
    output_repo: str,
    convert_to_hf: bool = False,
) -> None:
    base_model_dir = Path("checkpoints")
    adapter_dir = Path("adapter")
    merged_dir = Path("merged")

    download_from_hub(repo_id=base_model_repo, checkpoint_dir=base_model_dir)
    download_from_hub(repo_id=adapter_repo, checkpoint_dir=adapter_dir)

    lora_path = adapter_dir / "lit_model_lora_finetuned.pth"
    merge_lora(
        lora_path=Path(lora_path),
        checkpoint_dir=base_model_dir,
        out_dir=merged_dir,
    )

    for path in Path(base_model_dir).glob("*.json"):
        destination = Path(merged_dir) / path.name

        shutil.copy(src=path, dst=destination)

    if convert_to_hf:
        output_dir = Path("hf_checkpoint_merged")
        convert_lit_checkpoint(
            checkpoint_path=merged_dir / "lit_model.pth",
            output_path=output_dir,
            config_path=merged_dir / "lit_config.json",
        )
    else:
        output_dir = merged_dir

    upload_folder(repo_id=output_repo, folder_path=output_dir)
