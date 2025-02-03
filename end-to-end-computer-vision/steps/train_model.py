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

from typing import Annotated, Any, Dict, Tuple

from ultralytics import YOLO
from zenml import ArtifactConfig, log_metadata, step
from zenml.logger import get_logger

from materializers.label_studio_export_materializer import (
    LabelStudioAnnotationExport,
)
from materializers.ultralytics_materializer import UltralyticsMaterializer
from utils.dataset_utils import load_and_split_data
from zenml.enums import ArtifactType
logger = get_logger(__name__)


@step(
    output_materializers={"Trained_YOLO": UltralyticsMaterializer},
    enable_cache=True,
)
def train_model(
    model: YOLO,
    dataset: LabelStudioAnnotationExport,
    data_source: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    is_quad_gpu_env: bool = False,
    is_single_gpu_env: bool = False,
    is_apple_silicon_env: bool = False,
) -> Tuple[
    Annotated[
        YOLO, ArtifactConfig(name="Trained_YOLO", artifact_type=ArtifactType.MODEL)
    ],
    Annotated[Dict[str, Any], "validation_metrics"],
    Annotated[Dict[str, Any], "model_names"],
]:
    """Trains a model on a dataset.

    Args:
        epochs: Number of epochs to train the model for.
        batch_size: Batch size for training
        imgsz: Image size for training
        model: YOLO model to train.
        dataset: Dataset to train the model on.
        data_source: Source where the data lives
        is_quad_gpu_env: Whether we are in an env with 4 gpus
        is_single_gpu_env: Whether we are in an env with a single gpu
        is_apple_silicon_env: In case we are running on Apple compute

    Returns:
        Tuple[YOLO, Dict[str, Any]]: Trained model and validation metrics.
    """
    logger.info(f"Training YOLO model on dataset {dataset}")
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    logger.info("Loading and splitting data...")
    data_path = load_and_split_data(dataset=dataset, data_source=data_source)

    logger.info("Training model...")
    if is_quad_gpu_env:
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=[0, 1, 2, 3],
        )
    elif is_single_gpu_env:
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=[0],
        )
    elif is_apple_silicon_env:
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device="mps",
        )
    else:
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
        )

    logger.info("Evaluating model...")
    metrics = model.val()  # evaluate model performance on the validation set
    
    log_metadata(
        artifact_name="Trained_YOLO",
        infer_artifact=True,
        metadata={"metrics": metrics.results_dict, "names": model.names},
    )
    
    return model, metrics.results_dict, model.names
