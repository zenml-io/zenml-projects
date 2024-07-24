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
from datasets import load_from_disk
from materializers.directory_materializer import DirectoryMaterializer
from mlflow.models import infer_signature
from utils.loaders import load_pretrained_model
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
        is_eval=True,
    )
    model = load_pretrained_model(
        ft_model_dir,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    test_dataset = load_from_disk(str((datasets_dir / "test_raw").absolute()))
    sample = test_dataset[1]
    
    # MLflow infers schema from the provided sample input/output/params
    signature = infer_signature(
        model_input=sample["instruction"],
        model_output=sample["output"],
        # Parameters are saved with default values if specified
        params={"max_new_tokens": 100, "do_sample": True , "temperature": 0.5, "top_p": 0.5, "pad_token_id": tokenizer.eos_token_id},
    )
    prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n"
        "### user: {prompt}\n"
        "### assistant:"
    )

    
    # If you interrupt the training, uncomment the following line to stop the MLflow run
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tokenizer},
        prompt_template=prompt,
        signature=signature,
        artifact_path="model",  # This is a relative path to save model files within MLflow run
    )
    
    # register mlflow model
    mlflow_register_model_step.entrypoint(
        model,
        name=model_name,
    )
    # keep track of mlflow version for future use
    model_registry = Client().active_stack.model_registry
    if model_registry:
        version = model_registry.list_model_versions(
            name=model_name, stage=None, order_by_date="version", limit=1
        )[0]
        if version:
            model_ = get_step_context().model
            model_.log_metadata({"model_registry_version": version.version})