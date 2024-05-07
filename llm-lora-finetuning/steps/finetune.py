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

from pathlib import Path

from functions.finetune import finetune_fn
from materializers.directory_materializer import DirectoryMaterializer
from typing_extensions import Annotated
from zenml import logging as zenml_logging
from zenml import step
from zenml.integrations.accelerate.utils.accelerate_runner import (
    run_with_accelerate,
)
from zenml.logger import get_logger
from zenml.materializers import BuiltInMaterializer
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)
zenml_logging.STEP_LOGS_STORAGE_MAX_MESSAGES = (
    10000  # workaround for https://github.com/zenml-io/zenml/issues/2252
)

cache_invalidator = hash(finetune_fn.__code__)


@step(output_materializers=[DirectoryMaterializer, BuiltInMaterializer])
def finetune(
    base_model_id: str,
    dataset_dir: Path,
    max_steps: int = 1000,
    logging_steps: int = 50,
    eval_steps: int = 50,
    save_steps: int = 50,
    optimizer: str = "paged_adamw_8bit",
    lr: float = 2.5e-5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 5,
    bf16: bool = True,
    use_accelerate: bool = False,
    use_fast: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    cache_invalidator: int = cache_invalidator,
) -> Annotated[Path, "ft_model_dir"]:
    """Finetune the model using PEFT.

    Base model will be derived from configure step and finetuned model will
    be saved to the output directory.

    Finetuning parameters can be found here: https://github.com/huggingface/peft#fine-tuning

    Args:
        base_model_id: The base model id to use.
        dataset_dir: The path to the dataset directory.
        max_steps: The maximum number of steps to train for.
        logging_steps: The number of steps to log at.
        eval_steps: The number of steps to evaluate at.
        save_steps: The number of steps to save at.
        optimizer: The optimizer to use.
        lr: The learning rate to use.
        per_device_train_batch_size: The batch size to use for training.
        gradient_accumulation_steps: The number of gradient accumulation steps.
        warmup_steps: The number of warmup steps.
        bf16: Whether to use bf16.
        use_accelerate: Whether to use accelerate.
        use_fast: Whether to use the fast tokenizer.
        load_in_4bit: Whether to load the model in 4bit mode.
        load_in_8bit: Whether to load the model in 8bit mode.

    Returns:
        The path to the finetuned model directory.
    """
    cleanup_gpu_memory(force=True)

    ft_model_dir = "model_dir"
    common_kwargs = dict(
        base_model_id=base_model_id,
        dataset_dir=dataset_dir.as_posix(),
        max_steps=max_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        optimizer=optimizer,
        lr=lr,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        bf16=bf16,
        use_fast=use_fast,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        use_accelerate=use_accelerate,
        ft_model_dir=ft_model_dir,
    )
    if not use_accelerate:
        finetune_fn(
            **common_kwargs,
        )
    else:
        run_with_accelerate(
            function=finetune_fn, label_names=["input_ids"], **common_kwargs
        )

    return Path(ft_model_dir)
