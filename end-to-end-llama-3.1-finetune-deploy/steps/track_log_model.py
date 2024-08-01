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
from typing import Any, Dict, Optional

import mlflow
import torch
from datasets import load_from_disk
from materializers.directory_materializer import DirectoryMaterializer
from mlflow.models import infer_signature
from peft import PeftModel
from transformers import pipeline
from trl import setup_chat_format
from utils.loaders import load_base_model
from utils.tokenizer import load_tokenizer
from zenml import get_step_context, step
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
def track_log_model(
    base_model_id: str,
    system_prompt: str,
    datasets_dir: Path,
    model_name: str,
    ft_model_dir: Optional[Path],
    use_fast: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> None:
    """Track and log the model to Databricks.
    
    Args:
        base_model_id: The base model id to use.
        system_prompt: The system prompt to use.
        datasets_dir: The path to the datasets directory.
        model_name: The name of the model to log.
        ft_model_dir: The path to the finetuned model directory. If None, the
            base model will be used.
        use_fast: Whether to use the fast tokenizer.
        load_in_4bit: Whether to load the model in 4bit mode.
        load_in_8bit: Whether to load the model in 8bit mode.
        
    Returns:
        None
    """
    cleanup_gpu_memory(force=True)
    tokenizer = load_tokenizer(
        base_model_id,
        is_eval=False,
        use_fast=False,
    )
    base_model = load_base_model(
        base_model_id,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    model = PeftModel.from_pretrained(base_model, ft_model_dir)
    model = model.merge_and_unload()
    
    # register mlflow model
    mlflow.set_registry_uri("databricks-uc")
    #mlflow.set_experiment(f"/Users/7c2a45bf-fd61-46b7-a0a3-6ff6d7b81a7a/{get_step_context().pipeline.name}")

    
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
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "max_length": 512,
    }
    signature = mlflow.models.infer_signature(
        input,
        mlflow.transformers.generate_signature_output(pipe, input, params=params),
        params=params,
    )
    #outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # MLflow infers schema from the provided sample input/output/params
    #signature = infer_signature(
    #    model_input=sample["Patient"],
    #    model_output=sample["Doctor"],
    #    # Parameters are saved with default values if specified
    #    params={"max_new_tokens": 256, "do_sample": True , "temperature": 0.7, "top_p": 0.95, "top_k": 50, "pad_token_id": tokenizer.eos_token_id},
    #)
    prompt = (
        "You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.\n"
        "User: {prompt}\n"
        "Assistant:"
    )

    #prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    
    # If you interrupt the training, uncomment the following line to stop the MLflow run
    mlflow.transformers.log_model(
        transformers_model=pipe,
        prompt_template=prompt,
        signature=signature,
        artifact_path="model",  # This is a relative path to save model files within MLflow run
        registered_model_name=model_name,
        pip_requirements = ["transformers>=4.43.2", "peft", "bitsandbytes>=0.41.3", "zenml>=0.62.0", "vllm>=0.5.3.post1", "mlflow>=2.14.3", "accelerate>=0.30.0", "torchvision"]
    )
    
    # keep track of mlflow version for future use
    client = mlflow.MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % f"databricks_workspace.default.{model_name}")
    new_model_version = max([model_version_info.version for model_version_info in model_version_infos])
    model_ = get_step_context().model
    model_.log_metadata({"model_registry_version": new_model_version})