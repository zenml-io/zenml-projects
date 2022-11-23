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
from typing import Dict, Tuple

import cv2
import torch
from yolov5.detect import main, parse_opt
from zenml.steps import BaseParameters, step


class DetectParameters(BaseParameters):
    """Trainer params"""

    imgsz: Tuple = (768, 1024)
    conf: float = 0.5
    weights: str = "./inference/model/best.pt"
    source: str = "./inference/images/"


@step(enable_cache=True)
def detector(
    test_set: Dict,
    model: Dict,
    params: DetectParameters,
) -> None:
    """Train a neural net from scratch to recognize MNIST digits return our
    model or the learner"""

    os.makedirs(os.path.join("inference", "images"), exist_ok=True)
    os.makedirs(os.path.join("inference", "model"), exist_ok=True)

    image_saver(test_set)
    model_saver(model)

    # Run main to start training
    opt = parse_opt()
    opt.weights = params.weights
    opt.imgsz = params.imgsz
    opt.conf_thres = params.conf
    opt.source = params.source
    main(opt)


def image_saver(image_set: Dict):
    for key, value in image_set.items():
        dim = (768, 1024)
        resized_image = cv2.resize(value[0], dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'inference/images/{key.rsplit("/",1)[1]}', resized_image)


def model_saver(model: Dict):
    torch.save(model, "./inference/model/best.pt")
