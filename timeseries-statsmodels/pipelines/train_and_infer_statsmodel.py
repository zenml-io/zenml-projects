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
    sarimax_trainer_step,
    cpi_data_loader_step
)

from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def train_and_infer_statsmodel(data_stream: str = None):
    """This pipeline trains an individual statsmodel."""
    # Load the data
    data = cpi_data_loader_step(data_stream)
    
    # Train the SARIMAX model
    model = sarimax_trainer_step(data=data)
