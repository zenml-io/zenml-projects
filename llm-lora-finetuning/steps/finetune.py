# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import shutil
from pathlib import Path
from typing import Optional

from finetune.lora import setup
from huggingface_hub import upload_folder
from lit_gpt.args import IOArgs
from zenml import step

from scripts.convert_hf_checkpoint import convert_hf_checkpoint
from scripts.convert_lit_checkpoint import convert_lit_checkpoint
from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora
from scripts.prepare_alpaca import prepare


@step
def finetune(
    repo_id: str,
    adapter_output_repo: Optional[str] = None,
    merged_output_repo: Optional[str] = None,
    convert_to_hf: bool = False,
    data_dir: Optional[Path] = None,
) -> None:
    checkpoint_dir = Path("checkpoints")
    output_dir = Path("out/lora/alpaca")
    download_from_hub(repo_id=repo_id, checkpoint_dir=checkpoint_dir)
    convert_hf_checkpoint(checkpoint_dir=checkpoint_dir)

    if not data_dir:
        data_dir = Path("data/alpaca")
        prepare(destination_path=data_dir, checkpoint_dir=checkpoint_dir)

    io_args = (
        IOArgs(
            train_data_dir=data_dir,
            val_data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            out_dir=output_dir,
        ),
    )
    setup(precision="bf16-true", io=io_args)

    model_name = repo_id.split("/")[-1]

    if merged_output_repo:
        lora_path = output_dir / model_name / "lit_model_lora_finetuned.pth"

        merge_output_dir = Path("out/lora_merged") / model_name
        merge_lora(
            lora_alpha=lora_path,
            checkpoint_dir=checkpoint_dir,
            out_dir=merge_output_dir,
        )

        for path in Path(checkpoint_dir).glob("*.json"):
            destination = Path(merge_output_dir) / path.name

            shutil.copy(src=path, dst=destination)

        if convert_to_hf:
            upload_dir = Path("hf_checkpoint_merged")
            convert_lit_checkpoint(
                checkpoint_path=merged_output_repo,
                output_path=output_dir,
                config_path=merged_output_repo / "lit_config.json",
            )
        else:
            upload_dir = merge_output_dir

        upload_folder(repo_id=merged_output_repo, folder_path=upload_dir)

    if adapter_output_repo:
        upload_folder(
            repo_id=adapter_output_repo, folder_path=output_dir / model_name
        )
