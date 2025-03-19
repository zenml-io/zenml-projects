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

# steps/data_preprocessor.py
from datasets import Dataset, ClassLabel
from typing import Dict, Any
from zenml.steps import step


@step
def data_preprocessor(dataset: Dataset) -> Dataset:
    """
    Standardizes text inputs and label encoding.

    Args:
        dataset: Raw input dataset

    Returns:
        Dataset: Processed dataset with:
            - Combined title + text fields
            - Binary labels (0: negative, 1: positive)
            - ClassLabel schema for labels

    Note:
        Handles multiple label formats:
        - String labels: "accept"/"good_case_study"/"positive" -> 1
        - Binary labels: Preserves existing 0/1 encoding
    """

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        example["text"] = f"{example.get('meta_title', '')}. {example.get('text', '')}"

        if "answer" in example and isinstance(example["answer"], str):
            example["label"] = 1 if example["answer"].strip().lower() == "accept" else 0
        elif "label" in example and isinstance(example["label"], str):
            example["label"] = (
                1
                if example["label"].strip().lower() in ["good_case_study", "positive", "accept"]
                else 0
            )

        return example

    processed = dataset.map(preprocess)
    processed = processed.cast_column("label", ClassLabel(names=["negative", "positive"]))

    return processed
