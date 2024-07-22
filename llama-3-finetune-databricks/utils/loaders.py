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
from typing import Any, Dict, Optional, Tuple, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, TrainingArguments
from zenml.client import Client

from utils.logging import print_trainable_parameters


def get_lora_config() -> LoraConfig:
    """Get the Lora configuration.

    Returns:
        The Lora configuration.
    """
    config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.08,  # Conventional
        task_type="CAUSAL_LM",
    )
    return config
     
def get_create_trainer_args(
    output_dir: str,
    warmup_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    eval_steps: int,
    max_steps: int,
    bf16: bool,
    optimizer: str,
    learning_rate: float,
    logging_steps: Optional[int],
    save_steps: Optional[int],
    gradient_checkpointing: Optional[bool] = False,
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    save_strategy: str = "steps",
    evaluation_strategy: str = "steps",
    do_eval: bool = True,
    logging_dir: str = "./logs",
) -> TrainingArguments:
    """Get the arguments for creating the trainer.
    
    Args:
        output_dir: The output directory.
        warmup_steps: The number of warmup steps.
        per_device_train_batch_size: The batch size per device.
        gradient_checkpointing: Whether to use gradient checkpointing.
        gradient_checkpointing_kwargs: The gradient checkpointing kwargs.
        gradient_accumulation_steps: The number of gradient accumulation steps.
        max_steps: The maximum number of steps.
        learning_rate: The learning rate.
        logging_steps: The number of logging steps.
        bf16: Whether to use bfloat16.
        optimizer: The optimizer to use.
        save_strategy: The save strategy.
        save_steps: The number of save steps.
        evaluation_strategy: The evaluation strategy.
        eval_steps: The number of evaluation steps.
        do_eval: Whether to perform evaluation.
        logging_dir: The logging directory.
    
    Returns:
        The arguments for creating the trainer.
    """
    return TrainingArguments(
        output_dir=output_dir,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=(
            min(logging_steps, max_steps) if max_steps >= 0 else logging_steps
        ),
        bf16=bf16,
        optim=optimizer,
        logging_dir=logging_dir,
        save_strategy=save_strategy,
        save_steps=min(save_steps, max_steps) if max_steps >= 0 else save_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        do_eval=do_eval,
        ddp_find_unused_parameters=False,
        num_train_epochs=1,
    )
    
def load_base_model(
    base_model_id: str,
    lora_config: LoraConfig,
    is_training: bool = True,
    use_accelerate: bool = False,
    should_print: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_flash_attention2: Optional[bool] = False,
) -> Union[Any, Tuple[Any, Dataset, Dataset]]:
    """Load the base model.

    Args:
        base_model_id: The base model id to use.
        lora_config: The Lora configuration.
        is_training: Whether the model is in training mode.
        use_accelerate: Whether to use accelerate.
        should_print: Whether to print the trainable parameters.
        load_in_8bit: Whether to load the model in 8-bit mode.
        load_in_4bit: Whether to load the model in 4-bit mode.

    Returns:
        The base model.
    """
    from accelerate import Accelerator
    from huggingface_hub import login
    from transformers import BitsAndBytesConfig
    secret = Client().get_secret("hf_token")
    login(token = secret.secret_values["token"])
    if use_accelerate:
        accelerator = Accelerator()
        device_map = {"": accelerator.process_index}
    else:
        device_map = {"": torch.cuda.current_device()}

    # Replace attention with flash attention 
    if torch.cuda.get_device_capability()[0] >= 8:
        use_flash_attention2 = True

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2" if use_flash_attention2 else "sdpa",
        token=secret.secret_values["token"],
    )

    if is_training:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        model = get_peft_model(model, lora_config)
        if should_print:
            print_trainable_parameters(model)
        if use_accelerate:
            model = accelerator.prepare_model(model)

    return model


def load_pretrained_model(
    ft_model_dir: Path,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> AutoModelForCausalLM:
    """Load the finetuned model saved in the output directory.

    Args:
        ft_model_dir: The path to the finetuned model directory.
        load_in_4bit: Whether to load the model in 4-bit mode.
        load_in_8bit: Whether to load the model in 8-bit mode.

    Returns:
        The finetuned model.
    """
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        ft_model_dir, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2",
    )
    return model
