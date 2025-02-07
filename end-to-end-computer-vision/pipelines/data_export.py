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
from steps.export_label_studio import load_data_from_label_studio
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def data_export_pipeline(dataset_name: str = "polution"):
    """Loads data from Label studio.

    Args:
        dataset_name: Name of the dataset to load.
    """
    load_data_from_label_studio(dataset_name=dataset_name)
