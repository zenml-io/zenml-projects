# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import upload_folder
from pydantic import BaseModel
from zenml import log_model_metadata, step

from scripts.convert_lit_checkpoint import convert_lit_checkpoint
from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora
from steps.params import LoraParameters
from steps.utils import (
    convert_to_lit_checkpoint_if_necessary,
    get_huggingface_access_token,
)


class MergeParameters(BaseModel):
    base_model_repo: str
    adapter_repo: str
    output_repo: str
    convert_to_hf_checkpoint: bool = False

    precision: Optional[str] = None
    lora: LoraParameters = LoraParameters()


@step
def merge(config: MergeParameters) -> None:
    """Merge base model and LoRA adapter.

    Args:
        config: Configuration for this step.
    """
    access_token = get_huggingface_access_token()

    base_model_dir = Path("checkpoints")
    adapter_dir = Path("adapter")
    merged_dir = Path("merged")

    download_from_hub(
        repo_id=config.base_model_repo,
        checkpoint_dir=base_model_dir,
        access_token=access_token,
    )
    download_from_hub(
        repo_id=config.adapter_repo,
        checkpoint_dir=adapter_dir,
        access_token=access_token,
    )

    convert_to_lit_checkpoint_if_necessary(
        checkpoint_dir=base_model_dir / config.model_repo
    )

    lora_path = (
        adapter_dir / config.adapter_repo / "lit_model_lora_finetuned.pth"
    )
    merge_lora(
        lora_path=Path(lora_path),
        checkpoint_dir=base_model_dir / config.base_model_repo,
        out_dir=merged_dir,
        precision=config.precision,
        **config.lora.dict()
    )

    for path in Path(base_model_dir).glob("*.json"):
        destination = Path(merged_dir) / path.name

        shutil.copy(src=path, dst=destination)

    if config.convert_to_hf_checkpoint:
        output_dir = Path("lora_merged_hf")
        convert_lit_checkpoint(
            checkpoint_path=merged_dir / "lit_model.pth",
            config_path=merged_dir / "lit_config.json",
            output_path=output_dir,
        )
    else:
        output_dir = merged_dir

    commit = upload_folder(
        repo_id=config.output_repo, folder_path=output_dir, token=access_token
    )
    log_model_metadata(
        metadata={
            "merged_model_huggingface_commit_hash": commit.oid,
            "merged_model_huggingface_commit_url": commit.commit_url,
        }
    )