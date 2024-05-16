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
    evaluate_model,
    load_datasets,
    train_model,
    dummy_evaluate_model,
    dummy_load_datasets,
    dummy_train_model,
)
from zenml import pipeline

DATASET_NAME = "zenml/rag_qa_embedding_questions"
MODEL_PATH = "all-MiniLM-L6-v2"
NUM_EPOCHS = 30
WARMUP_STEPS = 0.1  # 10% of train data

DUMMY_DATASET_NAME = "embedding-data/sentence-compression"
# DUMMY_MODEL_PATH = "embedding-data/distilroberta-base-sentence-transformer"
DUMMY_MODEL_PATH = "all-MiniLM-L6-v2"
DUMMY_EPOCHS = 10

@pipeline
def finetune_embeddings() -> float:
    """Fine-tunes embeddings and evaluates the model."""
    train_dataset, test_dataset = load_datasets(DATASET_NAME)

    model = train_model(
        train_dataset,
        model_path=MODEL_PATH,
        num_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
    )

    evaluate_model(model, MODEL_PATH, test_dataset)

# TODO: FOR TESTING ONLY (remove after)
@pipeline
def dummy_finetune_embeddings() -> float:
    """Dummy Fine-tunes embeddings and evaluates the model."""
    train_dataset, test_dataset = dummy_load_datasets(DUMMY_DATASET_NAME)

    model = dummy_train_model(
        train_dataset,
        model_path=DUMMY_MODEL_PATH,
        num_epochs=DUMMY_EPOCHS,
        warmup_steps=WARMUP_STEPS,
    )

    dummy_evaluate_model(model, DUMMY_MODEL_PATH, test_dataset)
