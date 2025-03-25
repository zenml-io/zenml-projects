# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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
from typing import Annotated, Dict, Tuple

import huggingface_hub
import transformers
from datasets import Dataset
from schemas import TrainingConfig, zenml_project
from utils import (
    apply_docker_settings,
    calculate_memory_usage,
    calculate_prediction_costs,
    compute_classification_metrics,
    logger,
    measure_inference_latency,
)
from utils.model_loaders import load_base_model, load_tokenizer
from utils.remote_setup import determine_device
from zenml import ArtifactConfig, log_metadata, step
from zenml.client import Client
from zenml.enums import ArtifactType
from zenml.utils.cuda_utils import cleanup_gpu_memory


@step(
    model=zenml_project,
    enable_cache=False,
    settings=apply_docker_settings(step_name="finetune_modernbert"),
)
def finetune_modernbert(
    train_set: Dataset,
    validation_set: Dataset,
    test_set: Dataset,
    training_params: Dict,
    project: Dict,
    base_model: str,
    remote_execution: bool = False,
) -> Tuple[
    Annotated[
        transformers.ModernBertForSequenceClassification,
        ArtifactConfig(
            name="ft_model",
            artifact_type=ArtifactType.MODEL,
            tags=["finetuned", "model"],
        ),
    ],
    Annotated[
        transformers.PreTrainedTokenizerFast,
        ArtifactConfig(
            name="ft_tokenizer",
            artifact_type=ArtifactType.MODEL,
            tags=["finetuned", "tokenizer"],
        ),
    ],
]:
    """Fine-tunes ModernBERT classifier for article classification.

    Args:
        train_set: Training data split
        validation_set: Validation data split
        test_set: Test data split
        training_params: Training hyperparameters and settings
        project: ZenML project metadata for versioning
        base_model: Base model ID to fine-tune
        remote_execution: Whether to use remote-optimized settings

    Returns:
        Tuple[Model, Tokenizer]: Trained model and tokenizer
    """
    cleanup_gpu_memory(force=True)

    client = Client()
    if not os.getenv("HF_TOKEN"):
        try:
            hf_token = client.get_secret("hf_token").secret_values["token"]
            huggingface_hub.login(token=hf_token)
        except Exception as e:
            logger.warning(f"Error authenticating with Hugging Face: {e}")

    device = determine_device()

    tokenizer = load_tokenizer(base_model)

    def tokenize_function(examples):
        return {
            **tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                max_length=512,
            ),
            "labels": examples["label"],
        }

    tokenized_train = train_set.map(
        tokenize_function,
        batched=True,
        remove_columns=train_set.column_names,
    )
    tokenized_validation = validation_set.map(
        tokenize_function,
        batched=True,
        remove_columns=validation_set.column_names,
    )
    tokenized_test = test_set.map(
        tokenize_function,
        batched=True,
        remove_columns=test_set.column_names,
    )

    labels = train_set.features["label"].names
    label2id = {"negative": 0, "positive": 1}
    id2label = {0: "negative", 1: "positive"}

    logger.info(f"Loading model from {base_model}")
    model = load_base_model(
        base_model_id=base_model,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        device=device,
        remote_execution=remote_execution,
    )

    args_config = TrainingConfig.from_dict(training_params)

    # validate training arguments
    training_args = args_config.to_training_arguments()

    # configure trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=compute_classification_metrics,
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.0001,
            ),
        ],
    )

    # train model
    trainer.train()

    # log run configuration
    run_metadata = {
        "project": {
            "base_model": base_model,
            "name": project["name"],
            "version": project["version"],
            "tags": project.get("tags", []),
        },
        "training_config": {
            "batch_size": training_args.per_device_train_batch_size,
            "eval_batch_size": training_args.per_device_eval_batch_size,
            "learning_rate": training_args.learning_rate,
            "num_epochs": training_args.num_train_epochs,
            "warmup_ratio": training_args.warmup_ratio,
            "label_smoothing": training_args.label_smoothing_factor,
            "weight_decay": training_args.weight_decay,
            "bf16": training_args.bf16,
            "fp16": training_args.fp16,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "max_steps": training_args.max_steps,
            "eval_steps": training_args.eval_steps,
        },
        "dataset_stats": {
            "train_samples": len(train_set),
            "validation_samples": len(validation_set),
            "test_samples": len(test_set),
            "num_labels": len(labels),
        },
        "device_info": {
            "device": device,
            "remote_execution": remote_execution,
        },
    }
    log_metadata(metadata=run_metadata, infer_model=True)
    log_metadata(metadata={"training_config": run_metadata})

    # evaluate model
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_test)

    # calculate performance metrics
    performance_metrics = {k: float(v) for k, v in eval_metrics.items()}
    latency_metrics = measure_inference_latency(trainer, tokenized_test)
    inference_cost = calculate_prediction_costs(
        trainer, latency_metrics["inference_latency"]
    )
    memory_usage = calculate_memory_usage()

    performance_metrics.update(
        {
            **latency_metrics,
            "cost_per_1000_predictions": inference_cost,
            "memory_usage_mb": memory_usage,
        }
    )

    # log performance metrics
    log_metadata(metadata=performance_metrics, infer_model=True)
    log_metadata(metadata={"performance_metrics": performance_metrics})

    return model, tokenizer
