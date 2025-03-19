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

from pathlib import Path

from datasets import Dataset

from schemas import zenml_project
from zenml import step


@step(model=zenml_project)
def save_test_set(test_set: Dataset, artifact_path: str) -> str:
    """
    Saves the Hugging Face Dataset to disk and returns the path.
    """

    artifact_path = Path(artifact_path)
    artifact_path.mkdir(parents=True, exist_ok=True)
    test_set.save_to_disk(str(artifact_path))
    return str(artifact_path)
