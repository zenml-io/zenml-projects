# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from pathlib import Path




from pathlib import Path




from lit_gpt.args import IOArgs
from zenml import step

from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora
from scripts.prepare_alpaca import prepare
from finetune.lora import setup
import shutil

@step
def merge(checkpoint_dir: str, lora_path: str, out_dir: str) -> None:
    merge_lora(lora_alpha=Path(lora_path), checkpoint_dir=Path(checkpoint_dir), out_dir=Path(out_dir))

    for path in Path(checkpoint_dir).glob('*.json'):
        destination = Path(out_dir) / path.name

        shutil.copy(src=path, dst=destination)