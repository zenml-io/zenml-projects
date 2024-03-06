# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import json
import shutil
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

from evaluate.lm_eval_harness import run_eval_harness
from zenml import step, log_artifact_metadata, log_model_metadata

from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora
from steps.utils import get_huggingface_access_token


@step
def eval(
    model_repo: str, adapter_repo: Optional[str] = None
) -> Annotated[Dict[str, Any], "evaluation_results"]:
    access_token = get_huggingface_access_token()

    model_dir = Path("model")
    download_from_hub(
        repo_id=model_repo, checkpoint_dir=model_dir, access_token=access_token
    )

    if adapter_repo:
        adapter_dir = Path("adapter")
        merged_dir = Path("merged")

        download_from_hub(
            repo_id=adapter_repo, checkpoint_dir=adapter_dir, access_token=access_token
        )

        lora_path = adapter_dir / "lit_model_lora_finetuned.pth"
        merge_lora(
            lora_path=Path(lora_path),
            checkpoint_dir=model_dir,
            out_dir=merged_dir,
        )

        for path in Path(model_dir).glob("*.json"):
            destination = Path(merged_dir) / path.name

            shutil.copy(src=path, dst=destination)

        model_dir = merged_dir

    output_path = Path("output.json")
    run_eval_harness(checkpoint_dir=model_dir, save_filepath=output_path)

    with open(output_path, "r") as f:
        return json.load(f)
