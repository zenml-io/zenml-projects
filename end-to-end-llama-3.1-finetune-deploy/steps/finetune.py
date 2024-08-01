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

import os
from pathlib import Path

import huggingface_hub
import transformers
from datasets import load_from_disk
from materializers.directory_materializer import DirectoryMaterializer
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, setup_chat_format
from typing_extensions import Annotated
from utils.callbacks import ZenMLCallback
from utils.loaders import load_base_model
from utils.logging import print_trainable_parameters
from utils.tokenizer import load_tokenizer
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.materializers import BuiltInMaterializer
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)

experiment_tracker = None
if Client().active_stack.experiment_tracker:
    experiment_tracker = Client().active_stack.experiment_tracker.name

@step(experiment_tracker=experiment_tracker, output_materializers=[DirectoryMaterializer, BuiltInMaterializer])
def finetune(
    base_model_id: str,
    dataset_dir: Path,
    max_steps: int = 1000,
    logging_steps: int = 50,
    eval_steps: int = 50,
    save_steps: int = 50,
    optimizer: str = "paged_adamw_32bit",
    lr: float = 2e-4,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 10,
    bf16: bool = True,
    use_fast: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Annotated[Path, ArtifactConfig(name="ft_model_dir", is_model_artifact=True)]:
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
        use_fast: Whether to use the fast tokenizer.
        load_in_4bit: Whether to load the model in 4bit mode.
        load_in_8bit: Whether to load the model in 8bit mode.

    Returns:
        The path to the finetuned model directory.
    """
    cleanup_gpu_memory(force=True)
    
    # authenticate with Hugging Face for gated repos
    client = Client()

    if not os.getenv("HF_TOKEN"):
        try:
            hf_token = client.get_secret("hf_token").secret_values["token"]
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            logger.warning(f"Error authenticating with Hugging Face: {e}")

    ft_model_dir = Path("model_dir")
    dataset_dir = Path(dataset_dir)

    should_print = True

    project = "zenml-finetune"
    run_name = base_model_id + "-" + project
    output_dir = "./" + run_name

    if should_print:
        logger.info("Loading datasets...")
    tokenizer = load_tokenizer(
        base_model_id,  
        is_eval=False,
        use_fast=False,
    )
    tokenized_train_dataset = load_from_disk(str((dataset_dir / "train").absolute()))
    tokenized_val_dataset = load_from_disk(str((dataset_dir / "val").absolute()))

    if should_print:
        logger.info("Loading base model...")

    # Define the training arguments
    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ddp_find_unused_parameters=False,
        #max_steps=max_steps,
        learning_rate=lr,
        logging_steps=(
            min(logging_steps, max_steps) if max_steps >= 0 else logging_steps
        ),
        bf16=False,
        fp16=False,
        optim=optimizer,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=min(save_steps, max_steps) if max_steps >= 0 else save_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        group_by_length=True,
        num_train_epochs=3,
    )
    
    # Define the peft LoRa configuration
    peft_config = LoraConfig(
        r=16,
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
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
        
    model = load_base_model(
        base_model_id,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit, 
    )
    
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    if should_print:
        print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        peft_config=peft_config,
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )
        
    if should_print:
        logger.info("Training model...")
    trainer.train()

    if should_print:
        logger.info("Saving model...")

    ft_model_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    trainer.model.save_pretrained(ft_model_dir)

    return ft_model_dir
