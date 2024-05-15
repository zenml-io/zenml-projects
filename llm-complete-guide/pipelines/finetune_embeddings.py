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

from steps.finetune_embeddings import (
    create_test_examples,
    create_train_examples,
    evaluate_model,
    load_datasets,
    train_model,
)
from zenml import pipeline

DATASET_NAME = "zenml/rag_qa_embedding_questions"
MODEL_PATH = "embedding-data/distilroberta-base-sentence-transformer"
NUM_EPOCHS = 30
WARMUP_STEPS = 0.1  # 10% of train data


@pipeline
def finetune_embeddings() -> float:
    """Fine-tunes embeddings and evaluates the model.

    Returns:
        The average cosine similarity on the test set.
    """
    train_dataset, test_dataset = load_datasets(DATASET_NAME)

    train_examples = create_train_examples(train_dataset)

    model = train_model(
        train_examples,
        model_path=MODEL_PATH,
        num_epochs=NUM_EPOCHS,
    )

    test_examples = create_test_examples(test_dataset)

    evaluate_model(model, test_examples)
