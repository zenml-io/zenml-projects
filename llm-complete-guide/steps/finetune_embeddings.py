#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
import tempfile
from io import BytesIO
from typing import Annotated, Dict

import matplotlib.pyplot as plt
import torch
from constants import (
    DATASET_NAME_ARGILLA,
    DATASET_NAME_DISTILABEL,
    EMBEDDINGS_MODEL_ID_BASELINE,
    EMBEDDINGS_MODEL_ID_FINE_TUNED,
    EMBEDDINGS_MODEL_MATRYOSHKA_DIMS, SECRET_NAME,
)
from datasets import DatasetDict, concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from PIL import Image
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from zenml import ArtifactConfig, log_model_metadata, step
from zenml.client import Client
from zenml.utils.cuda_utils import cleanup_gpu_memory


@step
def prepare_load_data(
    use_argilla_annotations: bool = False,
) -> Annotated[DatasetDict, "full_dataset"]:
    """Load and prepare the dataset for training and evaluation."""
    if use_argilla_annotations:
        zenml_client = Client()
        annotator = zenml_client.active_stack.annotator
        if not annotator:
            raise RuntimeError("No annotator found in the active stack.")
        dataset = annotator.get_labeled_data(dataset_name=DATASET_NAME_ARGILLA)
    else:
        # Load dataset from the hub
        dataset: (
            DatasetDict | Dataset | IterableDatasetDict | IterableDataset
        ) = load_dataset(DATASET_NAME_DISTILABEL, split="train")
        # Add an id column to the dataset
        dataset = dataset.add_column("id", range(len(dataset)))

    # split dataset into a 10% test set
    dataset = dataset.train_test_split(test_size=0.1)

    return dataset


def get_evaluator(
    dataset: DatasetDict,
    model: SentenceTransformer,
) -> SequentialEvaluator:
    """Create a SequentialEvaluator for the given dataset and model."""
    temp_dir = tempfile.TemporaryDirectory()
    train_dataset_path = os.path.join(temp_dir.name, "train_dataset.json")
    test_dataset_path = os.path.join(temp_dir.name, "test_dataset.json")

    # save datasets to disk
    dataset["train"].to_json(train_dataset_path, orient="records")
    dataset["test"].to_json(test_dataset_path, orient="records")

    # load datasets
    test_dataset = load_dataset(
        "json", data_files=test_dataset_path, split="train"
    )
    train_dataset = load_dataset(
        "json", data_files=train_dataset_path, split="train"
    )
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Convert the datasets to dictionaries
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)
    queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )  # Our queries (qid => question)

    # Create a mapping of relevant document (1 in our case) for each query
    relevant_docs: Dict[str, list] = {q_id: [q_id] for q_id in queries}

    matryoshka_evaluators = [
        InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
        for dim in EMBEDDINGS_MODEL_MATRYOSHKA_DIMS
    ]

    return SequentialEvaluator(matryoshka_evaluators)


def evaluate_model(
    dataset: DatasetDict, model: SentenceTransformer
) -> Dict[str, float]:
    """Evaluate the given model on the dataset."""
    cleanup_gpu_memory(force=True)

    evaluator = get_evaluator(
        dataset=dataset,
        model=model,
    )
    results = evaluator(model)

    for dim in EMBEDDINGS_MODEL_MATRYOSHKA_DIMS:
        key = f"dim_{dim}_cosine_ndcg@10"
        print(f"{key}: {results[key]}")

    return results


@step
def evaluate_base_model(
    dataset: DatasetDict,
) -> Annotated[Dict[str, float], "base_model_evaluation_results"]:
    """Evaluate the base model on the given dataset."""
    model = SentenceTransformer(
        EMBEDDINGS_MODEL_ID_BASELINE,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = evaluate_model(
        dataset=dataset,
        model=model,
    )

    # Convert numpy.float64 values to regular Python floats
    # (needed for serialization)
    base_model_eval = {
        f"dim_{dim}_cosine_ndcg@10": float(
            results[f"dim_{dim}_cosine_ndcg@10"]
        )
        for dim in EMBEDDINGS_MODEL_MATRYOSHKA_DIMS
    }

    log_model_metadata(
        metadata={"base_model_eval": base_model_eval},
    )

    return results


@step
def evaluate_finetuned_model(
    dataset: DatasetDict,
) -> Annotated[Dict[str, float], "finetuned_evaluation_results"]:
    """Evaluate the finetuned model on the given dataset."""
    fine_tuned_model = SentenceTransformer(
        f"zenml/{EMBEDDINGS_MODEL_ID_FINE_TUNED}",
        device="cuda" if torch.cuda.is_available() else "cpu",
        revision="main",
    )

    results = evaluate_model(
        dataset=dataset,
        model=fine_tuned_model,
    )

    # Convert numpy.float64 values to regular Python floats
    # (needed for serialization)
    # Extract and log only the desired results
    finetuned_model_eval = {
        f"dim_{dim}_cosine_ndcg@10": float(
            results[f"dim_{dim}_cosine_ndcg@10"]
        )
        for dim in EMBEDDINGS_MODEL_MATRYOSHKA_DIMS
    }

    log_model_metadata(
        metadata={"finetuned_model_eval": finetuned_model_eval},
    )

    return results


@step
def finetune(
    dataset: DatasetDict,
    epochs: int = 4,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    optimizer: str = "adamw_torch_fused",
) -> Annotated[
    SentenceTransformer,
    ArtifactConfig(
        is_model_artifact=True,
        name="finetuned-model",
    ),
]:
    """Finetune the model on the given dataset."""
    cleanup_gpu_memory(force=True)

    # load model with SDPA for using Flash Attention 2
    model = SentenceTransformer(
        EMBEDDINGS_MODEL_ID_BASELINE,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"zenml/{EMBEDDINGS_MODEL_ID_FINE_TUNED}",
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model=model,
        loss=inner_train_loss,
        matryoshka_dims=EMBEDDINGS_MODEL_MATRYOSHKA_DIMS,
    )

    temp_dir = tempfile.TemporaryDirectory()
    train_dataset_path = os.path.join(temp_dir.name, "train_dataset.json")
    dataset["train"].to_json(train_dataset_path, orient="records")
    train_dataset = load_dataset(
        "json", data_files=train_dataset_path, split="train"
    )

    evaluator = get_evaluator(
        dataset=dataset,
        model=model,
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=EMBEDDINGS_MODEL_ID_FINE_TUNED,  # output directory and hugging face model ID
        num_train_epochs=epochs,  # number of epochs
        per_device_train_batch_size=batch_size,  # train batch size
        gradient_accumulation_steps=16,  # for a global batch size of 512
        per_device_eval_batch_size=16,  # evaluation batch size
        warmup_ratio=0.1,  # warmup ratio
        learning_rate=learning_rate,  # learning rate, 2e-5 is a good value
        lr_scheduler_type="cosine",  # use constant learning rate scheduler
        optim=optimizer,  # use fused adamw optimizer
        tf32=False,  # use tf32 precision
        bf16=False,  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="epoch",  # evaluate after each epoch
        save_strategy="epoch",  # save after each epoch
        logging_steps=10,  # log every 10 steps
        save_total_limit=3,  # save only the last 3 models
        load_best_model_at_end=True,  # load the best model when training ends
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
        report_to="none",  # turn off wandb tracking
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,  # training arguments
        train_dataset=train_dataset.select_columns(
            ["positive", "anchor"]
        ),  # training dataset
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()

    zenml_client = Client()

    trainer.model.push_to_hub(
        f"zenml/{EMBEDDINGS_MODEL_ID_FINE_TUNED}",
        exist_ok=True,
        token=zenml_client.get_secret(SECRET_NAME).secret_values["hf_token"]
    )

    log_model_metadata(
        metadata={
            "training_params": {
                "num_train_epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "base_model": EMBEDDINGS_MODEL_ID_BASELINE,
                "EMBEDDINGS_MODEL_MATRYOSHKA_DIMS": EMBEDDINGS_MODEL_MATRYOSHKA_DIMS,
                "optimizer": optimizer,
            },
            "hardware": {
                "accelerator": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU",
                "accelerator_type": [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else ["CPU"],
                "accelerator_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.0f}GB"
                if torch.cuda.is_available()
                else "N/A",
            },
        }
    )

    # handle materialization error with this workaround:
    # Save the model to a temporary file
    temp_dir = tempfile.TemporaryDirectory()
    temp_model_path = os.path.join(temp_dir.name, "model.pt")
    torch.save(trainer.model.state_dict(), temp_model_path)

    # Load the model from the temporary file
    rehydrated_model = SentenceTransformer(EMBEDDINGS_MODEL_ID_BASELINE)
    rehydrated_model.load_state_dict(torch.load(temp_model_path))

    # Clean up the temporary directory
    temp_dir.cleanup()

    return rehydrated_model


@step
def visualize_results(
    base_results: Dict[str, float],
    finetuned_results: Dict[str, float],
) -> Annotated[Image.Image, "evaluation_chart"]:
    """Visualize the results of the evaluation."""
    # extract out the key results only
    simple_finetuned_results = {
        f"dim_{dim}_cosine_ndcg@10": float(
            finetuned_results[f"dim_{dim}_cosine_ndcg@10"]
        )
        * 100
        for dim in EMBEDDINGS_MODEL_MATRYOSHKA_DIMS
    }
    simple_base_results = {
        f"dim_{dim}_cosine_ndcg@10": float(
            base_results[f"dim_{dim}_cosine_ndcg@10"]
        )
        * 100
        for dim in EMBEDDINGS_MODEL_MATRYOSHKA_DIMS
    }

    dimensions = list(simple_base_results.keys())
    base_values = list(simple_base_results.values())
    finetuned_values = list(simple_finetuned_results.values())

    y = range(len(dimensions))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [i - height / 2 for i in y],
        finetuned_values,
        height,
        label="Finetuned Model",
        color="red",
    )
    for i, v in enumerate(finetuned_values):
        ax.text(
            v - 1.5,
            i - height / 2,
            f"{v:.1f}",
            va="center",
            ha="right",
            color="white",
        )
    ax.barh(
        [i + height / 2 for i in y],
        base_values,
        height,
        label="Base Model",
        color="blue",
    )
    for i, v in enumerate(base_values):
        ax.text(
            v - 1.5,
            i + height / 2,
            f"{v:.1f}",
            va="center",
            ha="right",
            color="white",
        )

    ax.set_xlabel("Scores (%)")
    ax.set_title("Evaluation Results")
    ax.set_yticks(y)
    ax.set_yticklabels(dimensions)
    ax.set_xlim(0, 100)
    ax.legend()

    fig.tight_layout()

    # Convert the plot to a PIL Image
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    return img
