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

from typing import Annotated, Any, Dict, List, Tuple

import PIL
import torch
from constants import EVAL_BATCH_SIZE
from datasets import DownloadMode, load_dataset
from datasets.arrow_dataset import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader
from utils.visualization_utils import create_comparison_chart
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


def is_code_or_log(text: str, threshold: float = 0.03) -> bool:
    """Check if a text string contains a high proportion of special characters.

    Determine if a given text string contains a high proportion of special characters typical of code or logs.

    Args:
        text (str): The text to analyze.
        threshold (float): The proportion of special characters above which the text is considered code or logs.

    Returns:
        bool: True if the proportion of special characters in the text exceeds the threshold, otherwise False.
    """
    special_chars = set("{}[]()<>|&;*#$/")
    count = sum(1 for char in text if char in special_chars)
    return count / len(text) > threshold


@step
def load_datasets(
    dataset_name: str, threshold: float = 0.004
) -> Tuple[
    Annotated[Dataset, "train_dataset"],
    Annotated[Dataset, "test_dataset"],
]:
    """Load and filter the train and test datasets to exclude entries likely to be code or logs.

    Args:
        dataset_name: The name of the dataset to load.
        threshold: The threshold for determining if text is code or log.

    Returns:
        A tuple containing the filtered train and test datasets.
    """
    full_dataset = load_dataset(
        dataset_name, download_mode=DownloadMode.FORCE_REDOWNLOAD
    )

    # Assuming the dataset has a 'train' split, access it from the DatasetDict
    train_dataset = full_dataset["train"]

    # Split the train dataset into train and test
    train_test_split = train_dataset.train_test_split(test_size=0.3)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    print("train_dataset_length_raw", len(train_dataset))
    print("test_dataset_length_raw", len(test_dataset))

    log_artifact_metadata(
        artifact_name="train_dataset",
        metadata={"row_count": len(train_dataset)},
    )
    log_artifact_metadata(
        artifact_name="test_dataset",
        metadata={"row_count": len(test_dataset)},
    )

    return train_dataset, test_dataset
    # full_dataset = load_dataset(dataset_name)
    # # split the dataset into 70/30 train and test
    # # the splits don't exist in the dataset already
    # # so we have to do this manually
    # train_size = int(0.7 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_dataset, test_dataset = full_dataset.train_test_split(
    #     test_size=test_size, train_size=train_size
    # )

    # # train_dataset = load_dataset(dataset_name, split="train")
    # # test_dataset = load_dataset(dataset_name, split="test")
    # print("train_dataset_length_raw", len(train_dataset))
    # print("test_dataset_length_raw", len(test_dataset))

    # # # Filter datasets to remove entries that are likely to be code or logs
    # # filtered_train_dataset = train_dataset.filter(
    # #     lambda example, threshold=threshold: not is_code_or_log(
    # #         example["page_content"], threshold
    # #     )
    # # )
    # # filtered_test_dataset = test_dataset.filter(
    # #     lambda example, threshold=threshold: not is_code_or_log(
    # #         example["page_content"], threshold
    # #     )
    # # )
    # # print("filtered_train_dataset_length", len(filtered_train_dataset))
    # # print("filtered_test_dataset_length", len(filtered_test_dataset))

    # # return filtered_train_dataset, filtered_test_dataset
    # return train_dataset, test_dataset


def create_input_examples(train_data: Dataset) -> List[InputExample]:
    """Create InputExample instances for each page_content and generated_question pair.

    Args:
        train_data: The training dataset.

    Returns:
        A list of InputExample instances.
    """
    input_examples = []
    for example in train_data:
        page_content = example["page_content"]
        for question in example["generated_questions"].split("\n"):
            input_examples.append(InputExample(texts=[page_content, question]))
    return input_examples


@step(enable_step_logs=False)
def train_model(
    train_dataset: Dataset,
    model_path: str,
    num_epochs: int,
    warmup_steps: float,
) -> Annotated[SentenceTransformer, "trained_model"]:
    """Train the sentence transformer model using multiple GPUs.

    Args:
        train_dataset: The training dataset.
        model_path: The path to the pre-trained model.
        num_epochs: The number of training epochs.
        warmup_steps: The number of warmup steps.

    Returns:
        The trained sentence transformer model.
    """
    train_examples = create_input_examples(train_dataset)

    num_train_steps = len(train_examples) * num_epochs
    warmup_steps = int(num_train_steps * warmup_steps)

    # Initialize the model
    model = SentenceTransformer(model_path)

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        # Move the model to multiple GPUs
        model = torch.nn.DataParallel(model)

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=80 * num_gpus
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train the model using multiple GPUs
    model.module.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
    )

    log_artifact_metadata(
        artifact_name="trained_model",
        metadata={
            "model_path": model_path,
            "num_epochs": num_epochs,
            "warmup_steps": warmup_steps,
            "num_train_steps": num_train_steps,
            "num_gpus": num_gpus,
            "batch_size": 80 * num_gpus,
            "shuffle": True,
        },
    )

    return model.module


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Custom collate function for the DataLoader.

    Args:
        batch: A list of examples from the dataset.

    Returns:
        A tuple containing two lists:
            - question_texts: A list of question texts.
            - context_texts: A list of context texts.
    """
    question_texts = []
    context_texts = []
    for example in batch[0:4]:
        generated_questions = example["generated_questions"]
        for question in generated_questions:
            question_texts.append(question)
            context_texts.append(example["page_content"])

    return question_texts, context_texts


@step(enable_cache=False)
def evaluate_model(
    finetuned_model: SentenceTransformer,
    comparison_model: str,
    test_dataset: Dataset,
    num_generations: int,
) -> Tuple[
    Annotated[float, "pretrained_average_similarity"],
    Annotated[float, "finetuned_average_similarity"],
    Annotated[PIL.Image.Image, "evaluation_results"],
]:
    """Compare two models on the test set.

    Args:
        finetuned_model: The finetuned sentence transformer model.
        comparison_model: The path to the pretrained model.
        test_dataset: The dataset used for testing.

    Returns:
        A tuple containing the average cosine similarity for each model on the
        test set as well as an image visualising the comparison.
    """
    logger.info("Evaluating the finetuned model on the test set.")
    logger.info(f"Comparison model: {comparison_model}")
    logger.info(f"Number of test examples: {len(test_dataset)}")
    logger.info("Loading the pretrained model and test data.")

    pretrained_model = SentenceTransformer(comparison_model)
    # Utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        finetuned_model = torch.nn.DataParallel(finetuned_model)
        pretrained_model = torch.nn.DataParallel(pretrained_model)

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=collate_fn,
    )

    logger.info("Calculating average similarities for both models.")
    finetuned_avg_sim, pretrained_avg_sim = calculate_average_similarities(
        test_dataloader, finetuned_model, pretrained_model
    )

    logger.info("Creating a comparison plot.")
    comparison_plot = create_comparison_chart(
        ["Pretrained Model", "Finetuned Model"],
        pretrained_similarity=pretrained_avg_sim,
        finetuned_similarity=finetuned_avg_sim,
    )

    log_artifact_metadata(
        artifact_name="evaluation_results",
        metadata={
            "pretrained_average_similarity": {
                "value": pretrained_avg_sim,
                "unit": "cosine similarity",
            },
            "finetuned_average_similarity": {
                "value": finetuned_avg_sim,
                "unit": "cosine similarity",
            },
            "num_generations": num_generations,
            "comparison_model": comparison_model,
            "len_test_dataset": len(test_dataset),
            "eval_batch_size": EVAL_BATCH_SIZE,
            "shuffle": False,
        },
    )

    return (
        pretrained_avg_sim,
        finetuned_avg_sim,
        comparison_plot,
    )


def calculate_average_similarities(
    dataloader: DataLoader,
    finetuned_model: SentenceTransformer,
    pretrained_model: SentenceTransformer,
) -> Tuple[float, float]:
    """Calculate average cosine similarities for both models across all batches.

    Args:
        dataloader: DataLoader containing the test dataset.
        finetuned_model: The finetuned sentence transformer model.
        pretrained_model: The pretrained sentence transformer model.

    Returns:
        A tuple of average similarities for the finetuned and pretrained models.
    """
    finetuned_total_similarity = 0
    pretrained_total_similarity = 0
    num_examples = 0

    for batch in dataloader:
        question_texts, context_texts = batch
        finetuned_sim_scores, pretrained_sim_scores = (
            calculate_batch_similarities(
                question_texts,
                context_texts,
                finetuned_model,
                pretrained_model,
            )
        )

        finetuned_total_similarity += finetuned_sim_scores
        pretrained_total_similarity += pretrained_sim_scores
        num_examples += len(question_texts)

    finetuned_average_similarity = finetuned_total_similarity / num_examples
    pretrained_average_similarity = pretrained_total_similarity / num_examples

    return finetuned_average_similarity, pretrained_average_similarity


def calculate_batch_similarities(
    question_texts: List[str],
    context_texts: List[str],
    finetuned_model: SentenceTransformer,
    pretrained_model: SentenceTransformer,
) -> Tuple[float, float]:
    """Calculate cosine similarities for a single batch for both models.

    Args:
        question_texts: List of question texts from the batch.
        context_texts: List of context texts from the batch.
        finetuned_model: The finetuned sentence transformer model.
        pretrained_model: The pretrained sentence transformer model.

    Returns:
        A tuple of summed diagonal similarity scores for the finetuned and pretrained models.
    """
    # Access the underlying model using the 'module' attribute
    if isinstance(finetuned_model, torch.nn.DataParallel):
        finetuned_model = finetuned_model.module
    if isinstance(pretrained_model, torch.nn.DataParallel):
        pretrained_model = pretrained_model.module

    finetuned_question_embeddings = finetuned_model.encode(question_texts)
    finetuned_content_embeddings = finetuned_model.encode(context_texts)
    pretrained_question_embeddings = pretrained_model.encode(question_texts)
    pretrained_content_embeddings = pretrained_model.encode(context_texts)

    # Convert NumPy arrays to PyTorch tensors
    finetuned_question_embeddings = torch.from_numpy(
        finetuned_question_embeddings
    )
    finetuned_content_embeddings = torch.from_numpy(
        finetuned_content_embeddings
    )
    pretrained_question_embeddings = torch.from_numpy(
        pretrained_question_embeddings
    )
    pretrained_content_embeddings = torch.from_numpy(
        pretrained_content_embeddings
    )

    # Optimize cosine similarity calculation
    cosine_similarity = CosineSimilarity(dim=1)

    finetuned_similarity_scores = cosine_similarity(
        finetuned_question_embeddings, finetuned_content_embeddings
    )
    pretrained_similarity_scores = cosine_similarity(
        pretrained_question_embeddings, pretrained_content_embeddings
    )

    return (
        finetuned_similarity_scores.sum().item(),
        pretrained_similarity_scores.sum().item(),
    )


# @step
# def dummy_load_datasets(
#     dataset_name: str, threshold: float = 0.004
# ) -> Tuple[
#     Annotated[Dataset, "train_dataset"],
#     Annotated[Dataset, "test_dataset"],
# ]:
#     """Load and filter the train and test datasets to exclude entries likely to be code or logs.

#     Args:
#         dataset_name: The name of the dataset to load.
#         threshold: The threshold for determining if text is code or log.

#     Returns:
#         A tuple containing the filtered train and test datasets.
#     """
#     full_dataset = load_dataset(dataset_name, split="train")
#     full_dataset = full_dataset.shuffle()

#     train_test_split = 0.7
#     train_size = int(len(full_dataset) * train_test_split)

#     train_dataset = full_dataset.select(range(train_size))
#     test_dataset = full_dataset.select(range(train_size, len(full_dataset)))

#     return train_dataset, test_dataset


# @step(enable_step_logs=False)
# def dummy_train_model(
#     train_dataset: Dataset,
#     model_path: str,
#     num_epochs: int,
#     warmup_steps: float,
# ) -> Annotated[SentenceTransformer, "trained_model"]:
#     """Train the sentence transformer model using multiple GPUs.

#     Args:
#         train_dataset: The training dataset.
#         model_path: The path to the pre-trained model.
#         num_epochs: The number of training epochs.
#         warmup_steps: The number of warmup steps.

#     Returns:
#         The trained sentence transformer model.
#     """
#     # Initialize the model
#     model = SentenceTransformer(model_path)

#     train_examples = []
#     train_data = train_dataset["set"]
#     n_examples = len(train_dataset)

#     for i in range(n_examples):
#         example = train_data[i]
#         train_examples.append(InputExample(texts=[example[0], example[1]]))

#     train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
#     train_loss = losses.MultipleNegativesRankingLoss(model=model)

#     num_train_steps = len(train_dataloader) * num_epochs
#     warmup_steps = int(num_train_steps * warmup_steps)

#     # Train the model
#     model.fit(
#         train_objectives=[(train_dataloader, train_loss)],
#         epochs=num_epochs,
#         warmup_steps=warmup_steps,
#     )

#     return model


# def dummy_collate_fn(
#     batch: List[Dict[str, Any]],
# ) -> Tuple[List[str], List[str]]:
#     """
#     Custom collate function for the DataLoader.

#     Args:
#         batch: A list of examples from the dataset.

#     Returns:
#         A tuple containing two lists:
#             - question_texts: A list of question texts.
#             - context_texts: A list of context texts.
#     """
#     question_texts = [example["set"][0] for example in batch]
#     context_texts = [example["set"][1] for example in batch]
#     return question_texts, context_texts


# @step(enable_cache=False)
# def dummy_evaluate_model(
#     finetuned_model: SentenceTransformer,
#     comparison_model: str,
#     test_dataset: Dataset,
# ) -> Tuple[
#     Annotated[float, "pretrained_average_similarity"],
#     Annotated[float, "finetuned_average_similarity"],
#     Annotated[PIL.Image.Image, "comparison_plot"],
# ]:
#     """Compare two models on the test set.

#     Args:
#         finetuned_model: The finetuned sentence transformer model.
#         test_dataset: The test dataset.

#     Returns:
#         A tuple containing the average cosine similarity for each model on the
#         test set as well as an image visualising the comparison.
#     """
#     pretrained_model = SentenceTransformer(comparison_model)

#     test_dataloader = DataLoader(
#         test_dataset,
#         shuffle=False,
#         batch_size=32,
#         collate_fn=dummy_collate_fn,
#     )

#     finetuned_total_similarity = 0
#     pretrained_total_similarity = 0
#     num_examples = 0

#     for batch in test_dataloader:
#         question_texts, context_texts = batch
#         question_embeddings_finetuned = finetuned_model.encode(question_texts)
#         content_embeddings_finetuned = finetuned_model.encode(context_texts)
#         question_embeddings_pretrained = pretrained_model.encode(
#             question_texts
#         )
#         content_embeddings_pretrained = pretrained_model.encode(context_texts)

#         finetuned_similarity_scores = cosine_similarity(
#             question_embeddings_finetuned, content_embeddings_finetuned
#         )
#         pretrained_similarity_scores = cosine_similarity(
#             question_embeddings_pretrained, content_embeddings_pretrained
#         )

#         finetuned_total_similarity += (
#             finetuned_similarity_scores.diagonal().sum()
#         )
#         pretrained_total_similarity += (
#             pretrained_similarity_scores.diagonal().sum()
#         )
#         num_examples += len(question_texts)

#     finetuned_average_similarity = finetuned_total_similarity / num_examples
#     pretrained_average_similarity = pretrained_total_similarity / num_examples

#     comparison_plot = create_comparison_chart(
#         ["Pretrained Model", "Finetuned Model"],
#         pretrained_similarity=pretrained_average_similarity,
#         finetuned_similarity=finetuned_average_similarity,
#     )

#     return (
#         pretrained_average_similarity,
#         finetuned_average_similarity,
#         comparison_plot,
#     )
