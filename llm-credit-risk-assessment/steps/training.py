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

from functools import partial
import math
from typing import Any, Dict, List, Tuple
from zenml import step, save_artifact
from zenml.client import Client
from zenml.logger import get_logger
from schemas.configuration import Configuration
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    TrainerCallback
)
import transformers
import torch
from datasets import Dataset
from utils.samples import generate_and_tokenize_prompt
from utils.model import get_model_param_count
from transformers import Trainer

logger = get_logger(__name__)

def restore_checkpoint(config: Configuration)->Any:
    last_checkpoint = None
    if config.try_resume_from_checkpoint:
        try:
            response = Client().get_artifact_version("calm_checkpoints")
            if response is not None:
                last_checkpoint = response.load()
                logger.info("Restored from checkpoint.")
        except KeyError:
            logger.info("No checkpoint found.")
    return last_checkpoint

def preload_model(config: Configuration, last_checkpoint: Any)->Tuple[Any,Any]:
    if last_checkpoint:
        model = last_checkpoint
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=(
                getattr(torch, "float32")
                if not config.bf16 and config.no_fp16
                else getattr(torch, "float16") if not config.bf16 else getattr(torch, "bfloat16")
            ),
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference

    logger.info(
        "tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id),
    )
    logger.info(
        "tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id),
    )
    logger.info(
        "tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id),
    )
    return model,tokenizer

def apply_lora(config: Configuration, model:Any)->Any:
    # peft model
    if config.use_peft_lora:
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    return model

def prepare_data(config: Configuration, train_data: List[Dict[str,Any]], validation_data: List[Dict[str,Any]],tokenizer: Any)->Tuple[Dataset,Dataset]:
    train_data = Dataset.from_list(train_data)
    train_data = train_data.shuffle().map(
        partial(generate_and_tokenize_prompt, config.seq_length, tokenizer)
    )

    validation_data = Dataset.from_list(validation_data)
    validation_data = validation_data.shuffle().map(
        partial(generate_and_tokenize_prompt, config.seq_length, tokenizer)
    )

    for i in range(2):
        logger.info(
            "Eval tokenized example: {}".format(validation_data[i]),
        )
    for i in range(2):
        logger.info(
            "Train tokenized example: {}".format(train_data[i]),
        )

    return train_data, validation_data

class CheckpointCallback(TrainerCallback):

    def on_epoch_end(self, args, state, control, **kwargs):        
            model = kwargs["model"]
            save_artifact(model,name="calm_checkpoints")
            

@step
def fine_tune(
    train_data: List[Dict[str,Any]], validation_data: List[Dict[str,Any]], config: Configuration
):
    if config.should_log:
        transformers.utils.logging.set_verbosity_info()

    transformers.utils.logging.set_verbosity(logger.level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {config}")

    last_checkpoint = restore_checkpoint(config)

    # Set seed before initializing model.
    set_seed(config.seed)

    model,tokenizer = preload_model(config,last_checkpoint)

    model = apply_lora(config, model)

    if not config.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    train_data,validation_data = prepare_data(config, train_data, validation_data, tokenizer)
    
    training_nums = len(train_data)
    num_gpus = torch.cuda.device_count()

    logger.info(
        "num_gpus = {}, training_nums = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            num_gpus,
            training_nums,
            config.num_warmup_steps,
            config.eval_freq,
            config.save_freq,
        ),
    )
    logger.info(
        "val data nums = {}, training_nums = {}, batch_size = {}".format(
            len(validation_data), training_nums, config.batch_size
        ),
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="no",
        max_steps=config.max_steps,
        eval_steps=config.eval_freq,
        save_steps=config.save_freq,
        logging_steps=config.log_freq,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.num_warmup_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.no_gradient_checkpointing,
        fp16=not config.no_fp16,
        bf16=config.bf16,
        weight_decay=config.weight_decay,
        run_name=f"credit-risk-assessment",
        push_to_hub=config.push_to_hub,
        include_tokens_per_second=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    logger.info(
        f"Using {training_args.half_precision_backend} half precision backend",
    )
    # Train!
    len_dataloader = len(trainer.get_train_dataloader())
    num_update_steps_per_epoch = (
        len_dataloader // training_args.gradient_accumulation_steps
    )

    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    num_examples = trainer.num_examples(trainer.get_train_dataloader())
    num_train_samples = num_examples * training_args.num_train_epochs
    max_steps = math.ceil(
        training_args.num_train_epochs * num_update_steps_per_epoch
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(
        f"  Num train samples = {num_train_samples}",
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}",
    )
    logger.info(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}",
    )
    logger.info(
        f"  Total optimization steps = {max_steps}",
    )

    logger.info(
        f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True)}",
    )

    # https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/3
    model.config.use_cache = False

    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    logger.info(
        "\n Training completed!!! If there's a warning about missing keys above, please disregard :)",
    )
