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

from datetime import datetime

from constants import (
    EMBEDDINGS_MODEL_NAME_ZENML,
)
from steps.distilabel_generate_queries import generate_synthetic_queries
from steps.eval_pii import eval_pii
from steps.hf_dataset_loader import load_hf_dataset
from steps.push_to_argilla import push_to_argilla
from steps.push_to_hf import push_to_hf
from zenml import Model, pipeline

model_definition = Model(
    name=EMBEDDINGS_MODEL_NAME_ZENML,
    license="Apache",
    description="A fine-tuned embeddings model for ZenML documentation. Used for RAG retrieval.",
    use_cases="RAG retrieval",
    audience="ZenML users",
    tags=[
        "rag",
        "embeddings",
        "finetuning",
        "llm",
        "internal",
        "synthetic-data",
    ],
    limitations="Only works for ZenML documentation. Not generalizable to other domains. Entirely build with synthetic data. The data is also quite noisy on account of how the chunks were split.",
    trade_offs="Focused on a specific RAG retrieval use case. Not generalizable to other domains.",
    ethics="The data is entirely synthetic.",
    version=f"argilla-webinar-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
)


@pipeline(model=model_definition)
def generate_synthetic_data():
    train_dataset, test_dataset = load_hf_dataset()
    train_pii_results, test_pii_results = eval_pii(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
    train_with_queries, test_with_queries = generate_synthetic_queries(
        train_dataset=train_dataset, test_dataset=test_dataset
    )
    push_to_hf(
        train_dataset=train_with_queries,
        test_dataset=test_with_queries,
        after="eval_pii",
    )
    push_to_argilla(
        train_dataset=train_with_queries,
        test_dataset=test_with_queries,
        after="eval_pii",
    )


if __name__ == "__main__":
    generate_synthetic_data()
