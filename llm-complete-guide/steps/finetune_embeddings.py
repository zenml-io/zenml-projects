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
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from utils.visualization_utils import create_comparison_chart
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_datasets(
    dataset_name: str,
) -> Tuple[
    Annotated[Dataset, "train_dataset"],
    Annotated[Dataset, "test_dataset"],
]:
    """Load the train and test datasets.

    Args:
        dataset_name: The name of the dataset to load.

    Returns:
        A tuple containing the train and test datasets.
    """
    train_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")
    return train_dataset, test_dataset


@step(enable_step_logs=False)
def train_model(
    train_dataset: Dataset,
    model_path: str,
    num_epochs: int,
    warmup_steps: float,
) -> Annotated[SentenceTransformer, "trained_model"]:
    """Train the sentence transformer model using multiple GPUs.

    Args:
        train_examples: The training examples.
        model_path: The path to the pre-trained model.
        num_epochs: The number of training epochs.
        warmup_steps: The number of warmup steps.

    Returns:
        The trained sentence transformer model.
    """
    train_examples = {}
    train_data = train_dataset
    n_examples = train_dataset.num_rows

    for i in range(n_examples):
        example = train_data[i]
        train_examples[str(i)] = InputExample(
            texts=[example["generated_questions"][0], example["page_content"]]
        )

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
        list(train_examples.values()), shuffle=True, batch_size=32 * num_gpus
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train the model using multiple GPUs
    model.module.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
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
    question_texts = [example["generated_questions"][0] for example in batch]
    context_texts = [example["page_content"] for example in batch]
    return question_texts, context_texts


@step(enable_cache=False)
def evaluate_model(
    finetuned_model: SentenceTransformer,
    test_dataset: Dataset,
) -> Tuple[
    Annotated[float, "pretrained_average_similarity"],
    Annotated[float, "finetuned_average_similarity"],
    Annotated[PIL.Image.Image, "comparison_plot"],
]:
    """Compare two models on the test set.

    Args:
        finetuned_model: The finetuned sentence transformer model.
        test_dataset: The test dataset.

    Returns:
        A tuple containing the average cosine similarity for each model on the
        test set as well as an image visualising the comparison.
    """
    pretrained_model = SentenceTransformer(
        "embedding-data/distilroberta-base-sentence-transformer"
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=32,
        collate_fn=collate_fn,
    )

    finetuned_total_similarity = 0
    pretrained_total_similarity = 0
    num_examples = 0

    for batch in test_dataloader:
        question_texts, context_texts = batch
        question_embeddings_finetuned = finetuned_model.encode(question_texts)
        content_embeddings_finetuned = finetuned_model.encode(context_texts)
        question_embeddings_pretrained = pretrained_model.encode(
            question_texts
        )
        content_embeddings_pretrained = pretrained_model.encode(context_texts)

        finetuned_similarity_scores = cosine_similarity(
            question_embeddings_finetuned, content_embeddings_finetuned
        )
        pretrained_similarity_scores = cosine_similarity(
            question_embeddings_pretrained, content_embeddings_pretrained
        )

        finetuned_total_similarity += (
            finetuned_similarity_scores.diagonal().sum()
        )
        pretrained_total_similarity += (
            pretrained_similarity_scores.diagonal().sum()
        )
        num_examples += len(question_texts)

    finetuned_average_similarity = finetuned_total_similarity / num_examples
    pretrained_average_similarity = pretrained_total_similarity / num_examples

    comparison_plot = create_comparison_chart(
        ["Pretrained Model", "Finetuned Model"],
        [pretrained_average_similarity, finetuned_average_similarity],
    )

    return (
        finetuned_average_similarity,
        pretrained_average_similarity,
        comparison_plot,
    )
