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
from zenml import pipeline
from zenml.logger import get_logger

from steps.process_hf_dataset import process_hf_dataset
from steps.upload_to_label_studio import upload_labels_to_label_studio

logger = get_logger(__name__)


@pipeline
def data_ingestion_pipeline():
    labels_dict = process_hf_dataset()
    upload_labels_to_label_studio(labels_dict)
