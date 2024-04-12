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


from steps import configure, evaluate_model, finetune, prepare_data, promote
from zenml import logging as zenml_logging
from zenml import pipeline
from zenml.config import DockerSettings

zenml_logging.STEP_LOGS_STORAGE_MAX_MESSAGES = (
    10000  # workaround for https://github.com/zenml-io/zenml/issues/2252
)


@pipeline(
    settings={
        "docker": DockerSettings(
            parent_image="pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime",
            requirements="requirements.txt",
        )
    }
)
def llm_peft_full_finetune():
    system_prompt, base_model_id = configure()
    datasets_dir = prepare_data(
        base_model_id=base_model_id, system_prompt=system_prompt
    )
    ft_model_dir = finetune(
        base_model_id,
        datasets_dir,
    )
    evaluate_model(
        base_model_id,
        system_prompt,
        datasets_dir,
        ft_model_dir,
        id="evaluate_finetuned",
    )
    evaluate_model(
        base_model_id,
        system_prompt,
        datasets_dir,
        None,
        id="evaluate_base",
    )
    promote(after=["evaluate_finetuned", "evaluate_base"])
