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
#
from typing import Optional

from steps.finetune import finetune_lora
from zenml import pipeline
from zenml.config import DockerSettings


@pipeline(settings={"docker": DockerSettings(requirements="requirements.txt")})
def finetuning_pipeline(
    repo_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
    adapter_output_repo: Optional[str] = None,
    merged_output_repo: Optional[str] = None,
    convert_to_hf: bool = False,
    data_dir: Optional[str] = None,
) -> None:
    finetune_lora(
        repo_id=repo_id,
        adapter_output_repo=adapter_output_repo,
        merged_output_repo=merged_output_repo,
        convert_to_hf=convert_to_hf,
        data_dir=data_dir
    )
