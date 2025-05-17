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


from steps import (
    evaluate_model,
    finetune,
    log_metadata_from_step_artifact,
    prepare_data,
    promote,
)
from zenml import pipeline
from zenml.integrations.huggingface.steps import run_with_accelerate


@pipeline
def llm_peft_full_finetune(
    system_prompt: str,
    base_model_id: str,
    use_fast: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    max_train_samples: int = None,
    max_val_samples: int = None,
    max_test_samples: int = None,
):
    """Pipeline for finetuning an LLM with peft.

    It will run the following steps:

    - prepare_data: prepare the datasets and tokenize them
    - finetune: finetune the model
    - evaluate_model: evaluate the base and finetuned model
    - promote: promote the model to the target stage, if evaluation was successful

    Args:
        system_prompt: The system prompt to use.
        base_model_id: The base model id to use.
        use_fast: Whether to use the fast tokenizer.
        load_in_8bit: Whether to load in 8-bit precision (requires GPU).
        load_in_4bit: Whether to load in 4-bit precision (requires GPU).
        max_train_samples: Maximum number of training samples to use (for CPU or testing).
        max_val_samples: Maximum number of validation samples to use (for CPU or testing).
        max_test_samples: Maximum number of test samples to use (for CPU or testing).
    """
    if not load_in_8bit and not load_in_4bit:
        raise ValueError(
            "At least one of `load_in_8bit` and `load_in_4bit` must be True."
        )
    if load_in_4bit and load_in_8bit:
        raise ValueError(
            "Only one of `load_in_8bit` and `load_in_4bit` can be True."
        )

    datasets_dir = prepare_data(
        base_model_id=base_model_id,
        system_prompt=system_prompt,
        use_fast=use_fast,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )

    evaluate_model(
        base_model_id,
        system_prompt,
        datasets_dir,
        None,
        use_fast=use_fast,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        id="evaluate_base",
    )
    log_metadata_from_step_artifact(
        "evaluate_base",
        "base_model_rouge_metrics",
        after=["evaluate_base"],
        id="log_metadata_evaluation_base",
    )

    finetune_accelerated = run_with_accelerate(
        finetune, num_processes=2, multi_gpu=True, mixed_precision="bf16"
    )
    ft_model_dir = finetune_accelerated(
        base_model_id=base_model_id,
        dataset_dir=datasets_dir,
        use_fast=use_fast,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        use_accelerate=True,
    )

    evaluate_model(
        base_model_id,
        system_prompt,
        datasets_dir,
        ft_model_dir,
        use_fast=use_fast,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        id="evaluate_finetuned",
    )
    log_metadata_from_step_artifact(
        "evaluate_finetuned",
        "finetuned_model_rouge_metrics",
        after=["evaluate_finetuned"],
        id="log_metadata_evaluation_finetuned",
    )

    promote(
        after=[
            "log_metadata_evaluation_finetuned",
            "log_metadata_evaluation_base",
        ]
    )
