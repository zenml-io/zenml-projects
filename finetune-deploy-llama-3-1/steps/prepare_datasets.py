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
from functools import partial
from pathlib import Path
from typing import Dict

import huggingface_hub
from materializers.directory_materializer import DirectoryMaterializer
from typing_extensions import Annotated
from utils.tokenizer import (
    format_chat_template,
    load_tokenizer,
)
from zenml import log_model_metadata, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.materializers import BuiltInMaterializer
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)

@step(output_materializers=[DirectoryMaterializer, BuiltInMaterializer])
def prepare_data(
    base_model_id: str,
    system_prompt: str,
    dataset_name: str = "ruslanmv/ai-medical-chatbot",
) -> Annotated[Path, "datasets_dir"]:
    """Prepare the datasets for finetuning.

    Args:
        base_model_id: The base model id to use.
        system_prompt: The system prompt to use.
        dataset_name: The name of the dataset to use.

    Returns:
        The path to the datasets directory.
    """
    from datasets import load_dataset
    cleanup_gpu_memory(force=True)
    
        # authenticate with Hugging Face for gated repos
    client = Client()

    if not os.getenv("HF_TOKEN"):
        try:
            hf_token = client.get_secret("hf_token").secret_values["token"]
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            logger.warning(f"Error authenticating with Hugging Face: {e}")

    log_model_metadata(
        {
            "system_prompt": system_prompt,
            "base_model_id": base_model_id,
        }
    )
    
    # Load the tokenizer
    tokenizer = load_tokenizer(base_model_id)
    
    # Format the chat template
    formatted_chat_template = partial(format_chat_template, tokenizer=tokenizer)

    # Load the dataset
    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.shuffle(seed=63).select(range(20000))
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Extract the train dataset
    train_dataset = dataset["train"]
    tokenized_train_dataset = train_dataset.map(
        formatted_chat_template,
        num_proc=4,
    )
    
    # Extract the validation dataset
    eval_test_dataset = dataset["test"]
    eval_test_dataset = eval_test_dataset.train_test_split(test_size=0.1)
    eval_dataset = eval_test_dataset["train"]
    tokenized_eval_dataset = eval_dataset.map(
        formatted_chat_template,
        num_proc=4,
    )
    test_dataset = eval_test_dataset["test"]

    # Save the datasets
    datasets_path = Path("datasets")
    tokenized_train_dataset.save_to_disk(str((datasets_path / "train").absolute()))
    tokenized_eval_dataset.save_to_disk(str((datasets_path / "val").absolute()))
    test_dataset.save_to_disk(str((datasets_path / "test_raw").absolute()))

    return datasets_path
