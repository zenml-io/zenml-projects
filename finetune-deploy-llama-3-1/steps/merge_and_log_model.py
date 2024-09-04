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
from typing import Any, Dict, Optional

import huggingface_hub
import mlflow
import torch
from datasets import load_from_disk
from materializers.directory_materializer import DirectoryMaterializer
from mlflow.models import infer_signature
from mlflow.utils.timeout import run_with_timeout
from peft import PeftModel
from transformers import AutoModelForCausalLM, pipeline
from trl import setup_chat_format
from typing_extensions import Annotated
from utils.loaders import load_base_model
from utils.tokenizer import load_tokenizer
from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.mlflow.steps.mlflow_registry import (
    mlflow_register_model_step,
)
from zenml.logger import get_logger
from zenml.materializers import BuiltInMaterializer
from zenml.utils.cuda_utils import cleanup_gpu_memory

experiment_tracker = None
if Client().active_stack.experiment_tracker:
    experiment_tracker = Client().active_stack.experiment_tracker.name

logger = get_logger(__name__)


@step(experiment_tracker=experiment_tracker, output_materializers=[DirectoryMaterializer, BuiltInMaterializer])
def merge_and_log_model(
    base_model_id: str,
    model_name: str,
    ft_model_dir: Optional[Path],
) -> Annotated[Path, ArtifactConfig(name="merged_model_dir", is_model_artifact=True)]:
    """Log the merged model to the MLflow registry.
    
    Args:
        base_model_id: The base model id to use.
        model_name: The name of the model to log.
        ft_model_dir: The path to the finetuned model directory. If None, the
            base model will be used.
        
    Returns:
        None
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
    
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(
        base_model_id,
    )
    
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    model = PeftModel.from_pretrained(base_model, ft_model_dir)
    model = model.merge_and_unload()
    
    logger.info("Saving merged model...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    
    
    model_registry = Client().active_stack.model_registry
    
    if model_registry:
        # register mlflow model
        messages = [{"role": "user", "content": "I have bad acne. what i do?"}]
        input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        params = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_length": 512,
        }
        signature = mlflow.models.infer_signature(
            input,
            mlflow.transformers.generate_signature_output(pipe, input, params=params),
            params=params,
        )
        prompt = """User: {prompt}\n Assistant:""" 
        
        mlflow.transformers.log_model(
            await_registration_for=3600,
            transformers_model=pipe,
            prompt_template=prompt,
            signature=signature,
            artifact_path="model",  # This is a relative path to save model files within MLflow run
            registered_model_name=model_name,
            pip_requirements = ["transformers>=4.43.2", "peft", "bitsandbytes>=0.41.3", "vllm>=0.5.3.post1", "mlflow>=2.14.3", "accelerate>=0.30.0", "torchvision"]
        )
        
        # keep track of mlflow version for future use
        client = mlflow.MlflowClient()
        model_version_infos = client.search_model_versions("name = '%s'" % f"{model_name}")
        new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
        model_ = get_step_context().model
        model_.log_metadata({"model_registry_version": new_model_version})
    
    return merged_model_dir