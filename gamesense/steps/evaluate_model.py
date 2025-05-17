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
from utils.loaders import (
    load_base_model,
    load_pretrained_model,
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
    cpu_only: bool = False,
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
        cpu_only: Whether to force using CPU only and disable quantization.
    """
    # Force disable GPU optimizations if in CPU-only mode
    if cpu_only:
        load_in_4bit = False
        load_in_8bit = False

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
        is_eval=True,
        use_fast=use_fast,
    )
    test_dataset = load_from_disk(str((datasets_dir / "test_raw").absolute()))

    # Reduce dataset size for CPU evaluation to make it more manageable
    if cpu_only:
        logger.info("CPU-only mode: Using a smaller test dataset subset")
        test_dataset = test_dataset[:10]  # Use only 10 samples for CPU
    else:
        test_dataset = test_dataset[:50]  # Use 50 samples for GPU

    ground_truths = test_dataset["meaning_representation"]
    tokenized_train_dataset = tokenize_for_eval(
        test_dataset, tokenizer, system_prompt
    )

    if ft_model_dir is None:
        logger.info("Generating using base model...")
        model = load_base_model(
            base_model_id,
            is_training=False,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            cpu_only=cpu_only,
        )
    else:
        logger.info("Generating using finetuned model...")
        model = load_pretrained_model(
            ft_model_dir,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            cpu_only=cpu_only,
        )

    model.eval()

    # Adjust generation parameters for CPU
    max_new_tokens = 30 if cpu_only else 100

    # Preemptively disable use_cache for Phi models on CPU to avoid 'get_max_length' error
    is_phi_model = "phi" in base_model_id.lower()
    use_cache = not (is_phi_model and cpu_only)

    if not use_cache:
        logger.info("Preemptively disabling KV cache for Phi model on CPU")
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    with torch.no_grad():
        try:
            # Move inputs to the same device as the model
            device = next(model.parameters()).device
            input_ids = tokenized_train_dataset["input_ids"].to(device)
            attention_mask = tokenized_train_dataset["attention_mask"].to(
                device
            )

            # Generate with appropriate parameters
            logger.info(f"Generating with use_cache={use_cache}")
            predictions = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=2,
                use_cache=use_cache,  # Use the preemptively determined setting
                do_sample=False,  # Use greedy decoding for more stable results on CPU
            )
        except (AttributeError, RuntimeError) as e:
            logger.warning(
                f"Initial generation attempt failed with error: {str(e)}"
            )

            # First fallback: try with more safety settings
            if (
                "get_max_length" in str(e)
                or "DynamicCache" in str(e)
                or cpu_only
            ):
                logger.warning(
                    "Using fallback generation strategy with minimal parameters"
                )
                try:
                    # Force model to CPU if needed
                    if not str(next(model.parameters()).device) == "cpu":
                        logger.info("Moving model to CPU for generation")
                        model = model.to("cpu")

                    # Move inputs to CPU
                    input_ids = tokenized_train_dataset["input_ids"].to("cpu")
                    attention_mask = tokenized_train_dataset[
                        "attention_mask"
                    ].to("cpu")

                    predictions = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=20,  # Even smaller for safety
                        pad_token_id=2,
                        use_cache=False,  # Disable KV caching completely
                        do_sample=False,  # Use greedy decoding
                        num_beams=1,  # Simple beam search
                    )
                except (RuntimeError, Exception) as e2:
                    logger.warning(
                        f"Second generation attempt failed with error: {str(e2)}"
                    )

                    # Final fallback: process one sample at a time
                    logger.warning(
                        "Final fallback: processing one sample at a time"
                    )

                    # Process one sample at a time
                    all_predictions = []
                    batch_size = tokenized_train_dataset["input_ids"].shape[0]

                    for i in range(batch_size):
                        try:
                            # Process one sample at a time
                            single_input = tokenized_train_dataset[
                                "input_ids"
                            ][i : i + 1].to("cpu")
                            single_attention = tokenized_train_dataset[
                                "attention_mask"
                            ][i : i + 1].to("cpu")

                            single_pred = model.generate(
                                input_ids=single_input,
                                attention_mask=single_attention,
                                max_new_tokens=20,  # Even further reduced for safety
                                num_beams=1,
                                do_sample=False,
                                use_cache=False,
                                pad_token_id=2,
                            )
                            all_predictions.append(single_pred)
                        except Exception as sample_error:
                            logger.error(
                                f"Failed to generate for sample {i}: {str(sample_error)}"
                            )
                            # Create an empty prediction as placeholder
                            all_predictions.append(
                                tokenized_train_dataset["input_ids"][i : i + 1]
                            )

                    # Combine the individual predictions
                    if all_predictions:
                        predictions = torch.cat(all_predictions, dim=0)
                    else:
                        # If all samples failed, return original inputs
                        logger.error(
                            "All samples failed in generation. Using inputs as fallback."
                        )
                        predictions = tokenized_train_dataset["input_ids"]
            else:
                # Re-raise if not a cache-related issue
                raise e
    predictions = tokenizer.batch_decode(
        predictions[:, tokenized_train_dataset["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    logger.info("Computing ROUGE metrics...")
    prefix = "base_model_" if ft_model_dir is None else "finetuned_model_"
    rouge = evaluate.load("rouge")
    rouge_metrics = rouge.compute(
        predictions=predictions, references=ground_truths
    )

    logger.info("Computed metrics: " + str(rouge_metrics))

    save_artifact(rouge_metrics, prefix + "rouge_metrics")
