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
from zenml import pipeline, get_pipeline_context, step
from zenml.logger import get_logger

from steps.train_model import train_model
from steps.predict_image import predict_image
from steps.load_model import load_model
from zenml import Model
logger = get_logger(__name__)
from zenml.client import Client



@step
def predict_over_images():
    images_dir_path = "data/ships/subset"
    # model = Model(
    #     name="Yolo_Object_Detection",
    #     version="staging",
    # )
    # model.get_model_artifact(name="staging").load()
    # model_artifact = model_version.get_model_artifact(name="Raw_YOLO")
    artifact = Client().get_artifact_version('105c768c-8c86-465a-b018-b1a800ad4e19')
    yolo_model = artifact.load()
    results = yolo_model(images_dir_path, half=True, conf=0.6)
    breakpoint()


@pipeline
def inference():
    predict_over_images()
