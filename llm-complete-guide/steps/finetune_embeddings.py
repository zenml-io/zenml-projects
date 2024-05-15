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

from typing import Annotated, Dict, Tuple

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.metrics.pairwise import cosine_similarity
from structures import InputExampleDataset
from torch.utils.data import DataLoader
from zenml import step


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
def create_train_examples(
    train_dataset: Dataset,
) -> Annotated[Dict[str, InputExample], "train_examples"]:
    """Create training examples from the train dataset.

    Args:
        train_dataset: The train dataset.

    Returns:
        A dictionary mapping example indices to InputExample objects.
    """
    train_examples = {}
    train_data = train_dataset
    n_examples = train_dataset.num_rows

    for i in range(n_examples):
        example = train_data[i]
        train_examples[str(i)] = InputExample(
            texts=[example["generated_questions"][0], example["page_content"]]
        )
    return train_examples


@step
def train_model(
    train_examples: Dict[str, InputExample], model_path: str, num_epochs: int
) -> Annotated[SentenceTransformer, "trained_model"]:
    """Train the sentence transformer model.

    Args:
        train_examples: The training examples.
        model_path: The path to the pre-trained model.
        num_epochs: The number of training epochs.
        warmup_steps: The number of warmup steps.

    Returns:
        The trained sentence transformer model.
    """
    num_train_steps = len(train_examples) * num_epochs
    warmup_steps = int(num_train_steps * warmup_steps)

    model = SentenceTransformer(model_path)
    train_dataloader = DataLoader(
        list(train_examples.values()), shuffle=True, batch_size=32
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
    )
    return model


@step
def create_test_examples(
    test_dataset: Dataset,
) -> Annotated[Dict[str, InputExample], "test_examples"]:
    """Create test examples from the test dataset.

    Args:
        test_dataset: The test dataset.

    Returns:
        A dictionary mapping example indices to InputExample objects.
    """
    test_examples = {}
    test_data = test_dataset
    n_test_examples = test_dataset.num_rows

    for i in range(n_test_examples):
        example = test_data[i]
        test_examples[str(i)] = InputExample(
            texts=[example["generated_questions"][0], example["page_content"]]
        )
    return test_examples


@step
def evaluate_model(
    model: SentenceTransformer, test_examples: Dict[str, InputExample]
) -> Annotated[float, "average_similarity"]:
    """Evaluate the trained model on the test set.

    Args:
        model: The trained sentence transformer model.
        test_examples: The test examples.

    Returns:
        The average cosine similarity on the test set.
    """
    test_dataset = InputExampleDataset(list(test_examples.values()))
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    total_similarity = 0
    n_test_examples = len(test_examples)
    for batch in test_dataloader:
        question_embeddings = model.encode(batch[0])
        content_embeddings = model.encode(batch[1])
        similarity_scores = cosine_similarity(
            question_embeddings, content_embeddings
        )
        total_similarity += similarity_scores.diagonal().sum()

    average_similarity = total_similarity / n_test_examples
    return average_similarity
