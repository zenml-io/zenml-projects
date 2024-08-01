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
from utils.loaders import (
    load_base_model,
)
from utils.tokenizer import load_tokenizer, tokenize_for_eval
from zenml import save_artifact, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)


@step
def evaluate_model(
    base_model_id: str,
    system_prompt: str,
    datasets_dir: Path,
    ft_model_dir: Optional[Path],
    use_fast: bool = True,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> None:
    """Evaluate the model with ROUGE metrics.

    Args:
        base_model_id: The base model id to use.
        system_prompt: The system prompt to use.
        datasets_dir: The path to the datasets directory.
        ft_model_dir: The path to the finetuned model directory. If None, the
            base model will be used.
        use_fast: Whether to use the fast tokenizer.
        load_in_4bit: Whether to load the model in 4bit mode.
        load_in_8bit: Whether to load the model in 8bit mode.
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

    logger.info("Evaluating model...")

    logger.info("Loading dataset...")
    tokenizer = load_tokenizer(
        base_model_id,
        is_eval=False,
        use_fast=False,
    )
    test_dataset = load_from_disk(str((datasets_dir / "test_raw").absolute()))
    test_dataset = test_dataset[:50]
    ground_truths = test_dataset["Patient"]
    
    logger.info("Generating using base model...")
    base_model = load_base_model(
        base_model_id,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)

    if ft_model_dir:
        logger.info("Generating using finetuned model...")
        model = PeftModel.from_pretrained(base_model, ft_model_dir)
        model = model.merge_and_unload()
    else:
        model = base_model
    
    tokenized_test_dataset = tokenize_for_eval(ground_truths, tokenizer, system_prompt)
    predictions = model.generate(
        **tokenized_test_dataset,
        max_length=512, 
        #max_seq_length=512,
        #num_return_sequences=1,
        #max_new_tokens=200,
        temperature=0.7,
        num_return_sequences=1
        #top_k=50, 
        #top_p=0.95,
        #repetition_penalty=2.5,
    )
    predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )
        

    results = []
    for pred in predictions:
        logger.info("Generated response: " + pred)
        results.append(pred.split("assistant")[1])

    logger.info("Computing ROUGE metrics...")
    prefix = "base_model_" if ft_model_dir is None else "finetuned_model_"
    rouge = evaluate.load("rouge")
    rouge_metrics = rouge.compute(predictions=results, references=ground_truths)

    logger.info("Computed metrics: " + str(rouge_metrics))

    save_artifact(rouge_metrics, prefix + "rouge_metrics")
