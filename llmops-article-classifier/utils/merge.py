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

from typing import Dict, List, Tuple


def get_identifier(record: Dict) -> Tuple[str, str]:
    """Creates a unique identifier for a record based on URL and title."""
    return (record["meta"]["url"], record["meta"]["title"])


def transform_classification_results(results: Dict) -> List[Dict]:
    """
    Transforms classification results from the DeepSeek format
    to the composite dataset record format.

    Args:
        results: Raw classification results from DeepSeek model

    Returns:
        List of records in the dataset format
    """
    records = []

    valid_indices = [
        idx
        for idx in results["results"]["is_accepted"].keys()
        if "Validation Error" not in results["results"].get("reason", {}).get(idx, "")
    ]

    for idx in valid_indices:
        if not all(
            field.get(idx)
            for field in [results["results"]["meta_url"], results["results"]["meta_title"]]
        ):
            continue

        record = {
            "meta": {
                "url": results["results"]["meta_url"][idx],
                "title": results["results"]["meta_title"][idx],
                "published_date": results["results"]["meta_published_date"].get(idx, ""),
                "author": results["results"]["meta_author"].get(idx, ""),
            },
            "answer": "accept" if results["results"]["is_accepted"][idx] else "reject",
        }
        records.append(record)

    return records
