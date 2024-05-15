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

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
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


@step
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


@step
def evaluate_model(
    model: SentenceTransformer,
    test_dataset: Dataset,
) -> Annotated[float, "average_similarity"]:
    """Evaluate the trained model on the test set.

    Args:
        model: The trained sentence transformer model.
        test_dataset: The test dataset.

    Returns:
        The average cosine similarity on the test set.
    """
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=32,
        collate_fn=collate_fn,
    )

    total_similarity = 0
    num_examples = 0

    for batch in test_dataloader:
        question_texts, context_texts = batch
        question_embeddings = model.encode(question_texts)
        content_embeddings = model.encode(context_texts)
        similarity_scores = cosine_similarity(
            question_embeddings, content_embeddings
        )
        total_similarity += similarity_scores.diagonal().sum()
        num_examples += len(question_texts)

    average_similarity = total_similarity / num_examples
    return average_similarity
