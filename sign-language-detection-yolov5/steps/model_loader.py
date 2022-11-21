#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
from typing import Dict

import torch
from zenml.post_execution import get_pipeline
from zenml.steps import Output, step

from model import wrapped_model


@step
def model_loader() -> Output(model_path=str, model=torch.nn.Module):
    """Loads the trained models from previous training pipeline runs."""
    training_pipeline = get_pipeline("yolov5_pipeline")
    last_run = training_pipeline.runs[-1]
    model_path = "./inference/model/best.pt"

    try:
        model = last_run.get_step("trainer").output.read()
    except KeyError:
        print(
            f"Skipping {last_run.name} as it does not contain the trainer step"
        )
    model_saver(model, model_path)
    return model_path, wrapped_model(model_path)


def model_saver(model: Dict, model_path: str):
    """Saves the model to the local file system."""
    os.makedirs(os.path.join("inference", "model"), exist_ok=True)
    torch.save(model, model_path)
