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

import cv2
import numpy as np
import torch
from materializer.yolo_model_materializer import Yolov5ModelMaterializer
from yolov5.train import main, parse_opt
from zenml.client import Client
from zenml.logger import get_logger
from zenml.steps import BaseParameters, Output, step

logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

step_operator = Client().active_stack.step_operator

if not step_operator or not experiment_tracker:
    raise RuntimeError(
        "Your active stack needs to contain a step operator and an "
        "experiment tracker to run this pipeline."
    )


class TrainerParameters(BaseParameters):
    """Trainer params"""

    imgsz: int = 1024
    batch_size: int = 16
    epochs: int = 200


@step(
    enable_cache=True,
    step_operator=step_operator.name,
    experiment_tracker=experiment_tracker.name,
    output_materializers={"model": Yolov5ModelMaterializer},
)
def trainer(
    training_set: Dict,
    validation_set: Dict,
    params: TrainerParameters,
) -> Output(model=dict):
    """Train a neural net from scratch to recognize MNIST digits return our
    model or the learner"""

    # First we create new folder to save the data before start training
    for res in ["images", "labels"]:
        for folder in ["train", "valid"]:
            os.makedirs(os.path.join("augment", res, folder), exist_ok=True)

    # Save the training set in local path
    image_saver(training_set)
    # Save the validation set in local path
    image_saver(validation_set)

    opt = parse_opt(known=True)
    opt.imgsz = params.imgsz
    opt.batch_size = params.batch_size
    opt.epochs = params.epochs
    opt.exist_ok = True
    opt.data = "./yolov5/asl.yaml"
    opt.name = "ZenmlYolo"
    opt.weights = "yolov5m.pt"

    main(opt)

    # Load the best model and return it
    model = torch.load(
        os.path.join("yolov5", "runs", "train", opt.name, "weights", "best.pt")
    )
    return model


def image_saver(image_set: Dict):
    for key, value in image_set.items():
        cv2.imwrite(f"augment/images/{key}", value[0])
        np.savetxt(
            # Outputting .txt file to appropriate train/validation folders
            os.path.join(f"augment/labels/{key[:-4]}.txt"),
            np.array(value[1]),
            fmt=["%d", "%f", "%f", "%f", "%f"],
        )
