# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import importlib
from pathlib import Path

from zenml import step

from scripts.download import download_from_hub


@step
def feature_engineering(model_repo: str, dataset_name: str) -> None:
    checkpoint_dir = Path("checkpoints")
    download_from_hub(
        repo_id=model_repo, tokenizer_only=True, checkpoint_dir=checkpoint_dir
    )

    destination_dir = Path("data") / dataset_name

    helper_module = importlib.import_module(f"scripts/prepare_{dataset_name}")
    prepare_function = getattr(helper_module, "prepare")

    prepare_function(
        checkpoint_dir=checkpoint_dir, destination_path=destination_dir
    )
