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
import sys

import bentoml
from bentoml.io import NumpyNdarray, Text, JSON

sys.path.insert(0, "yolov5")

yolo_runner = bentoml.pytorch.get("sign_language_yolov5").to_runner()

svc = bentoml.Service(
    name="sign_language_yolov5_service",
    runners=[yolo_runner],
)


@svc.api(input=Text(), output=Text())
async def predict(img: str) -> str:
    assert isinstance(img, str)
    sign, _ = await yolo_runner.async_run(img)
    return sign


@svc.api(input=Text(), output=JSON())
async def predict_img(img: str) -> str:
    assert isinstance(img, str)
    _, img = await yolo_runner.async_run(img)
    return img
