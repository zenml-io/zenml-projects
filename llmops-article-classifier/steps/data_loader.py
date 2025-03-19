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

from typing import List

from datasets import Dataset, load_dataset

from schemas import InputArticle, zenml_project
from zenml import step


@step
def load_classification_dataset(dataset_path: str) -> List[InputArticle]:
    """Loads articles based on classification purpose.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        List[InputArticle]: Parsed articles ready for classification
    """
    dataset = load_dataset("json", data_files=dataset_path)["train"]

    def select_fields(example):
        return {
            "text": example["text"],
            "meta": {
                "url": example["meta"]["url"],
                "title": example["meta"]["title"],
                "published_date": example["meta"]["published_date"],
                "author": example["meta"]["author"],
            },
        }

    dataset = dataset.map(select_fields, remove_columns=dataset.column_names)
    return [InputArticle(**example) for example in dataset]


@step(model=zenml_project)
def load_training_dataset(dataset_path: str) -> Dataset:
    """Loads combined dataset of verified and model-classified articles.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Dataset: Preprocessed dataset with binary labels for training
    """
    dataset = load_dataset("json", data_files=dataset_path)["train"]

    def select_fields(example):
        return {
            "text": example["text"],
            "meta_title": example["meta"]["title"],
            "meta_url": example["meta"]["url"],
            "meta_published_date": example["meta"]["published_date"],
            "meta_author": example["meta"]["author"],
            "answer": example["answer"],
            "label": 1 if example["answer"].lower() == "accept" else 0,
        }

    return dataset.map(select_fields, remove_columns=dataset.column_names)
