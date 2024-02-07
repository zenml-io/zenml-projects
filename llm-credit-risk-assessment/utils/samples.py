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

# from itertools import chain
from typing import Any, Dict, List
# import pudb
import copy
from transformers import PreTrainedTokenizer
# import json

IGNORE_INDEX = -100


def generate_and_tokenize_prompt(
    model_max_length: int,
    tokenizer: PreTrainedTokenizer,
    data_point: Dict[str, Any],
    fix_length=False,
    padding_side="left",
):
    input_ids = []
    labels = []
    source = data_point["conversations"]
    for sentence in source:
        sentence_from = sentence["from"].lower()
        sentence_value = (
            "Human: \n" + sentence["value"] + "\n\nAssistant: \n"
            if sentence_from == "human"
            else sentence["value"]
        )  # https://github.com/LianjiaTech/BELLE/issues/337
        # conversation += sentence_value
        sentence_ids = tokenizer.encode(
            sentence_value, add_special_tokens=False
        )  # do not add bos_token_id
        label = (
            copy.deepcopy(sentence_ids)
            if sentence_from != "human"
            else [IGNORE_INDEX] * len(sentence_ids)
        )
        input_ids += sentence_ids
        labels += label
        # add eos at every end of assistant sentence
        if sentence_from != "human":
            input_ids += [tokenizer.eos_token_id]  # make sure eos_token_id is correct
            labels += [tokenizer.eos_token_id]

    input_ids = input_ids[:model_max_length]
    labels = labels[:model_max_length]

    if all(x == IGNORE_INDEX for x in labels):
        labels[18:24] = input_ids[
            18:24
        ]  # labels can not have all values being -100. 18 and 24 are just random numbers
    attention_mask = [1] * len(input_ids)

    if fix_length:
        if padding_side == "left":
            input_ids = [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            ) + input_ids
            labels = [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            ) + labels
            attention_mask = [0] * (
                model_max_length - len(attention_mask)
            ) + attention_mask
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (
                model_max_length - len(input_ids)
            )
            labels = labels + [tokenizer.pad_token_id] * (
                model_max_length - len(labels)
            )
            attention_mask = attention_mask + [0] * (
                model_max_length - len(attention_mask)
            )

    tokenized_full_prompt = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return tokenized_full_prompt