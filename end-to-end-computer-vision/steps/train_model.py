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
from typing import Annotated, Tuple, Dict, Any

from ultralytics import YOLO
from zenml import ArtifactConfig, log_artifact_metadata, step

from materializers.label_studio_yolo_dataset_materializer import (
    LabelStudioYOLODataset,
)
from materializers.yolo_materializer import UltralyticsMaterializer
from utils.dataset_utils import load_and_split_data, load_images_from_folder


@step(output_materializers={"Trained_YOLO": UltralyticsMaterializer})
def train_model(
    model: YOLO,
    dataset: LabelStudioYOLODataset,
) -> Tuple[
     Annotated[YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)],
     Annotated[Dict[str, Any], "validation_metrics"]
     ]:
    data_path = load_and_split_data(dataset=dataset)
    model.train(data=data_path, epochs=1)

    # model.train(data="coco8.yaml", epochs=3)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    log_artifact_metadata(
        artifact_name="Trained_YOLO",
        metadata={"metrics": metrics.results_dict},
    )

    # Read images as PIL images from directory metrics.save_dir for all png and jpg files
    images = load_images_from_folder(metrics.save_dir)

    return model, metrics.results_dict