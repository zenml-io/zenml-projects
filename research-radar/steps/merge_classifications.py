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

import json

from utils import (
    get_identifier,
    logger,
    transform_classification_results,
)
from zenml import step


@step
def merge_classifications(
    results_path: str,
    training_dataset_path: str = "data/composite_dataset.jsonl",
    augmented_dataset_path: str = "data/augmented_dataset.jsonl",
    source_dataset_path: str = "data/unclassified_dataset.jsonl",
):
    """
    Merges newly classified articles with the existing training dataset.
    Creates the augmented dataset file with combined data.

    Args:
        results_path: Path to the classification results JSON
        training_dataset_path: Path to the existing training dataset (composite)
        augmented_dataset_path: Path where the augmented dataset will be saved
        source_dataset_path: Path to the source dataset containing article text
    """
    with open(results_path, "r") as f:
        classification_results = json.load(f)

    classified_records = transform_classification_results(
        classification_results
    )

    source_articles_text = {}
    with open(source_dataset_path, "r") as f:
        for line in f:
            article = json.loads(line)
            url = article["meta"]["url"]
            source_articles_text[url] = article["text"]

    for record in classified_records:
        url = record["meta"]["url"]
        if url in source_articles_text:
            record["text"] = source_articles_text[url]

    training_records = []
    training_identifiers = set()

    with open(training_dataset_path, "r") as f:
        for line in f:
            record = json.loads(line)
            identifier = get_identifier(record)
            training_identifiers.add(identifier)
            training_records.append(record)

    records_to_add = []
    for record in classified_records:
        identifier = get_identifier(record)
        if identifier not in training_identifiers:
            if "text" in record:
                records_to_add.append(record)
            else:
                logger.log_warning(
                    f"Skipping record without text: {record['meta']['url']}"
                )

    with open(augmented_dataset_path, "w") as f:
        for record in training_records + records_to_add:
            f.write(json.dumps(record) + "\n")

    logger.log_output_file(
        augmented_dataset_path,
        f"Augmented Dataset (added {len(records_to_add)} new articles)",
    )
