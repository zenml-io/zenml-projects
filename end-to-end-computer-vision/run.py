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

from zenml import step, pipeline, Model, ArtifactConfig
from ultralytics import YOLO
from typing_extensions import Annotated
from zenml.logger import get_logger
from typing import Tuple

logger = get_logger(__name__)


@step
def load_model(model_checkpoint: str = "yolov8l.pt") -> Annotated[YOLO, ArtifactConfig(name="Raw_YOLO", is_model_artifact=True)]:
    return YOLO(model_checkpoint)


@step
def train_model(
    model: YOLO,
    dataset_name: str = "quickstart",
) -> Tuple[
    Annotated[YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)],
    Annotated[dict, "metrics"],
    Annotated[dict, "result"],
]:


    model.train(data='data.yaml', epochs=1)

    # model.train(data="coco8.yaml", epochs=3)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    return model, metrics, results


@pipeline(enable_cache=False, model=Model(name="Yolo_Object_Detection"))
def my_pipeline():
    model = load_model()
    train_model(model=model)

if __name__ == "__main__":
    my_pipeline()
