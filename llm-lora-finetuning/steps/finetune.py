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

import subprocess
from pathlib import Path

import torch
from materializers.directory_materializer import DirectoryMaterializer
from typing_extensions import Annotated
from utils.cuda import cleanup_memory
from zenml import logging as zenml_logging
from zenml import step
from zenml.logger import get_logger
from zenml.materializers import BuiltInMaterializer

from scripts.finetune import accelerated_finetune

logger = get_logger(__name__)
zenml_logging.STEP_LOGS_STORAGE_MAX_MESSAGES = (
    10000  # workaround for https://github.com/zenml-io/zenml/issues/2252
)


@step(output_materializers=[DirectoryMaterializer, BuiltInMaterializer])
def finetune(
    base_model_id: str,
    dataset_dir: Path,
    finetune_script_sha: str,
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
) -> Annotated[Path, "ft_model_dir"]:
    """Finetune the model using PEFT.

    Base model will be derived from configure step and finetuned model will
    be saved to the output directory.

    Finetuning parameters can be found here: https://github.com/huggingface/peft#fine-tuning

    Args:
        base_model_id: The base model id to use.
        dataset_dir: The path to the dataset directory.
        finetune_script_sha: The sha of the finetune script.
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
    cleanup_memory()
    if not use_accelerate:
        return accelerated_finetune(
            base_model_id=base_model_id,
            dataset_dir=dataset_dir,
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
            use_accelerate=False,
            use_fast=use_fast,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

    else:
        logger.info("Starting accelerate training job...")
        ft_model_dir = "model_dir"
        command = (
            f"accelerate launch --num_processes {torch.cuda.device_count()} "
        )
        command += str(Path("scripts/finetune.py").absolute()) + " "
        command += f'--base-model-id "{base_model_id}" '
        command += f'--dataset-dir "{dataset_dir}" '
        command += f"--max-steps {max_steps} "
        command += f"--logging-steps {logging_steps} "
        command += f"--eval-steps {eval_steps} "
        command += f"--save-steps {save_steps} "
        command += f"--optimizer {optimizer} "
        command += f"--lr {lr} "
        command += (
            f"--per-device-train-batch-size {per_device_train_batch_size} "
        )
        command += (
            f"--gradient-accumulation-steps {gradient_accumulation_steps} "
        )
        command += f"--warmup-steps {warmup_steps} "
        if bf16:
            command += f"--bf16 "
        if use_accelerate:
            command += f"--use-accelerate "
            command += f"-l input_ids "
            command += f'--ft-model-dir "{ft_model_dir}" '
        if use_fast:
            command += f"--use-fast "
        if load_in_4bit:
            command += f"--load-in-4bit "
        if load_in_8bit:
            command += f"--load-in-8bit "

        print(command)

        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        for stdout_line in result.stdout.split("\n"):
            print(stdout_line)
        if result.returncode == 0:
            logger.info("Accelerate training job finished.")
            return Path(ft_model_dir)
        else:
            logger.error(
                f"Accelerate training job failed. With return code {result.returncode}."
            )
            raise subprocess.CalledProcessError(result.returncode, command)
