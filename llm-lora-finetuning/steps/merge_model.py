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
from typing import Annotated, Any, Dict, Optional

import huggingface_hub
import torch
from materializers.directory_materializer import DirectoryMaterializer
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM
from utils.loaders import (
    load_base_model,
    load_pretrained_model,
)
from utils.tokenizer import load_tokenizer
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.materializers import BuiltInMaterializer
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)


@step(output_materializers=[DirectoryMaterializer, BuiltInMaterializer])
def merge_adapter_base_model(
    model_name: str,
    base_model_id: str,
    ft_model_dir: Optional[Path],
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Annotated[Path, ArtifactConfig(name="merged_model_dir", is_model_artifact=True)]:
    """Merge the base model with the finetuned model.
    
    Args:
        base_model_id: The base model id to use.
        model_name: The name of the model to log.
        ft_model_dir: The path to the finetuned model directory. If None, the
            base model will be used.
        load_in_4bit: Whether to load the model in 4bit mode.
        load_in_8bit: Whether to load the model in 8bit mode.
    
    Returns:
        The path to the merged model directory.
    """
    # authenticate with Hugging Face for gated repos
    client = Client()

    if not os.getenv("HF_TOKEN"):
        try:
            hf_token = client.get_secret("hf_token").secret_values["token"]
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            logger.warning(f"Error authenticating with Hugging Face: {e}")
            
    # Define the output directory
    merged_model_dir = Path("merged_model_dir")
    cleanup_gpu_memory(force=True)
    
    logger.info("Generating using base model...")
    base_model_reload = load_base_model(
        base_model_id,
        is_training=False,
        is_merging=True,
        use_accelerate=False,
        should_print=True,
        load_in_4bit=False,
        load_in_8bit=False,
    )
    
    logger.info("Generating using finetuned model...")
    
    logger.info("Merging models...")
    model = PeftModel.from_pretrained(base_model_reload, ft_model_dir)
    model = model.merge_and_unload()

    logger.info("Saving merged model...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_model_dir)
    
    # Load the tokenizer
    tokenizer = load_tokenizer(
        base_model_id,
        is_eval=True,
        use_fast=False,
    )
    tokenizer.save_pretrained(merged_model_dir)
    
    return merged_model_dir
    
    
    
    