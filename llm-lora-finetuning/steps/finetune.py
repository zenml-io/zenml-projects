# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from pathlib import Path
from typing import Tuple, Annotated

from lit_gpt.args import IOArgs
from zenml import step

from scripts.download import download_from_hub
from scripts.convert_hf_checkpoint import convert_hf_checkpoint
from scripts.prepare_alpaca import prepare
from finetune.lora import setup
from scripts.merge_lora import merge_lora
import shutil

@step
def finetune_lora(repo_id: str) -> Tuple[Annotated[str, "checkpoint_dir"], Annotated[str, "output_path"]]:
    checkpoint_dir = Path("checkpoints")
    data_dir = Path("data/alpaca")
    output_dir = Path("out/lora/alpaca")
    download_from_hub(repo_id=repo_id, checkpoint_dir=checkpoint_dir)
    convert_hf_checkpoint(checkpoint_dir=checkpoint_dir)
    prepare(destination_path=data_dir, checkpoint_dir=checkpoint_dir)
    
    io_args = IOArgs(
        train_data_dir=data_dir,
        val_data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        out_dir=output_dir,
    ),
    setup(precision="bf16-true", io=io_args)

    model_name = repo_id.split("/")[-1]
    lora_path = output_dir / model_name / "lit_model_lora_finetuned.pth"

    merge_output_dir = Path("out/lora_merged") / model_name
    merge_lora(lora_alpha=lora_path, checkpoint_dir=checkpoint_dir, out_dir=merge_output_dir)

    for path in Path(checkpoint_dir).glob('*.json'):
        destination = Path(merge_output_dir) / path.name

        shutil.copy(src=path, dst=destination)

    return checkpoint_dir, lora_path
