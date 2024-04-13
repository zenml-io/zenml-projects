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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoTokenizer


def load_tokenizer(
    base_model_id: str,
    is_eval: bool = False,
) -> "AutoTokenizer":
    from transformers import AutoTokenizer

    if is_eval:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, add_bos_token=True, device_map="auto"
        )
        tokenizer.pad_token_id = 0
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True,
            device_map="auto",
        )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize(
    prompt: str,
    tokenizer: "AutoTokenizer",
) -> dict:
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(
    data_point: dict,
    tokenizer: "AutoTokenizer",
    system_prompt: str,
):
    full_prompt = f"""{system_prompt}

### Target sentence:
{data_point["target"]}

### Meaning representation:
{data_point["meaning_representation"]}
"""
    return tokenize(full_prompt, tokenizer)


def tokenize_for_eval(
    data_points: dict,
    tokenizer: "AutoTokenizer",
    system_prompt: str,
):
    eval_prompts = [
        f"""{system_prompt}

### Target sentence:
{data_point}

### Meaning representation:
"""
        for data_point in data_points["target"]
    ]
    return tokenizer(eval_prompts, padding="longest", return_tensors="pt").to(
        "cuda"
    )