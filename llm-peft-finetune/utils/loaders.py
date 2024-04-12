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
from typing import Any
from functools import partial
from utils.tokenizer import load_tokenizer, generate_and_tokenize_prompt
from utils.logging import print_trainable_parameters
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def prepare_data(
    base_model_id: str,
    system_prompt: str,
    is_tain: bool = True,
    is_eval: bool = True,
    is_test: bool = False,
) -> Any:
    if not (is_tain or is_eval):
        raise ValueError("Please set is_tain or is_eval to True")

    tokenizer = load_tokenizer(base_model_id, False)
    gen_and_tokenize = partial(
        generate_and_tokenize_prompt, tokenizer=tokenizer, system_prompt=system_prompt
    )
    ret = []
    if is_tain:
        train_dataset = load_dataset("gem/viggo", split="train")
        tokenized_train_dataset = train_dataset.map(gen_and_tokenize)
        ret.append(tokenized_train_dataset)
    if is_eval:
        eval_dataset = load_dataset("gem/viggo", split="validation")
        tokenized_val_dataset = eval_dataset.map(gen_and_tokenize)
        ret.append(tokenized_val_dataset)
    if is_test:
        test_dataset = load_dataset("gem/viggo", split="test")
        tokenized_test_dataset = test_dataset.map(gen_and_tokenize)
        ret.append(tokenized_test_dataset)

    if len(ret) == 3:
        return tokenizer, ret[0], ret[1], ret[2]
    if len(ret) == 2:
        return tokenizer, ret[0], ret[1]
    return tokenizer, ret[0]


def load_base_model(base_model_id: str, is_training: bool = True) -> Any:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto"
    )

    if is_training:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    return model


def load_pretrained_model(ft_model_dir: Path) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        ft_model_dir, quantization_config=bnb_config, device_map="auto"
    )
    return model
