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
from typing import Optional

import evaluate
import huggingface_hub
import torch
from datasets import load_from_disk
from peft import PeftModel
from trl import setup_chat_format
from typing_extensions import Annotated
from utils.llama_cpp import process_model
from zenml import ArtifactConfig, save_artifact, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)


@step
def opitmize_model(
    merged_model_dir: Optional[Path],
    model_name: str,
) -> Annotated[Path, ArtifactConfig(name="merged_model_dir", is_model_artifact=True)]:
    """Evaluate the model with ROUGE metrics.

    Args:
        merged_model_dir: The path to the merged model directory.
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
            
    # Define the output directory
    optimized_model_dir = Path("optimized_model_dir")

    process_model(
        model_path=merged_model_dir,
        output_dir=optimized_model_dir,
        q_method="Q4_K_M",  # This is now the default
        model_name=model_name,
    )
    
    return optimized_model_dir
