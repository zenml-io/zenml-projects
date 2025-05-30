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
#

from steps import (
    mirror_repositories,
    prepare_dataset,
)
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def generate_code_dataset(dataset_id: str):
    """
    This pipeline generates the code dataset.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    mirror_directory = mirror_repositories()
    prepare_dataset(mirror_directory, dataset_id)
