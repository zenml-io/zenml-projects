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
from zenml import step, pipeline, Model, ArtifactConfig, log_artifact_metadata
from ultralytics import YOLO
from typing_extensions import Annotated
from zenml.logger import get_logger

from pipelines.data_export import data_export
from pipelines.training import training
from utils import load_images_from_folder, load_and_split_data
from materializers.yolo_materializer import UltralyticsMaterializer

logger = get_logger(__name__)


@step
def load_model(
    model_checkpoint: str = "yolov8l.pt",
) -> Annotated[YOLO, ArtifactConfig(name="Raw_YOLO", is_model_artifact=True)]:
    logger.info(f"Loading YOLO checkpoint {model_checkpoint}")
    return YOLO(model_checkpoint)
    # return YOLO()


# @step(output_materializers={"Trained_YOLO": UltralyticsMaterializer})
# def train_model(
#     model: YOLO,
#     dataset_name: str = "zenml_test_project",
# ) -> Tuple[
#     Annotated[
#         YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)
#     ],
#     Annotated[Image.Image, "confusion_matrix"],
# ]:


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
    # return model, images[0]


@step
def predict_image(model: YOLO):
    results = model("https://ultralytics.com/images/bus.jpg")
    print(results)


@pipeline(enable_cache=True, model=Model(name="Yolo_Object_Detection"))
def my_pipeline():
    model = load_model()
    trained_model = train_model(model=model)
    predict_image(trained_model)


if __name__ == "__main__":
    data_export() #.with_options(config_path="configs/data_export_alexej.yaml")()
    training()#.with
