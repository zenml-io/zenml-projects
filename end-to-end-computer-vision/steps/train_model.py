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
from typing import Annotated

from ultralytics import YOLO
from zenml import step, ArtifactConfig, log_artifact_metadata

from materializers.yolo_materializer import UltralyticsMaterializer
from utils import load_images_from_folder, load_and_split_data


@step(output_materializers={"Trained_YOLO": UltralyticsMaterializer})
def train_model(
    model: YOLO,
    dataset_name: str = "zenml_test_project",
) -> Annotated[
    YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)
]:
    data_path = load_and_split_data(dataset_name)
    model.train(data=data_path, epochs=1)

    # model.train(data="coco8.yaml", epochs=3)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    log_artifact_metadata(
        artifact_name="Trained_YOLO",
        metadata={"metrics": metrics.results_dict},
    )

    # Read images as PIL images from directory metrics.save_dir for all png and jpg files
    images = load_images_from_folder(metrics.save_dir)

    return model