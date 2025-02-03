#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
from typing import List

from rich import print as rich_print
from zenml.integrations.bentoml.services import BentoMLLocalDeploymentService
from zenml.steps import step


@step
def predictor(
    inference_data: List,
    service: BentoMLLocalDeploymentService,
) -> None:
    """Run an inference request against the BentoML prediction service.

    Args:
        service: The BentoML service.
        data: The data to predict.
    """

    service.start(timeout=10)  # should be a NOP if already started
    for img in inference_data:
        img = os.path.join(os.getcwd(), img)
        rich_print(service.predict("predict", img))
