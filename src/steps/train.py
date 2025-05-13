# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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

from datetime import datetime
from typing import Annotated, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from zenml import ArtifactConfig, log_metadata, step
from zenml.enums import ArtifactType
from zenml.logger import get_logger

from src.constants import (
    MODAL_MODEL_PATH,
    MODEL_NAME,
    TEST_DATASET_NAME,
    TRAIN_DATASET_NAME,
)
from src.utils import save_artifact_to_modal
from src.utils.model_definition import model_definition

logger = get_logger(__name__)


@step(model=model_definition)
def train_model(
    train_df: Annotated[pd.DataFrame, TRAIN_DATASET_NAME],
    test_df: Annotated[pd.DataFrame, TEST_DATASET_NAME],
    target: str = "target",
    hyperparameters: Optional[Dict] = None,
) -> Annotated[
    GradientBoostingClassifier,
    ArtifactConfig(name=MODEL_NAME, artifact_type=ArtifactType.MODEL),
]:
    """Train a GradientBoosting model.

    Logs hyper-params & model checksum (Annex IV §2 b,g).

    Args:
        train_df: Training dataset.
        test_df: Test dataset.
        target: Target column name.
        hyperparameters: Hyperparameters for the model.

    Returns:
        model: The trained GradientBoostingClassifier model
        model_path: Path to the model in Modal Volume
    """
    params = hyperparameters or {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
    }

    # data preprocessor set may have added a suffix to the target column
    target_col = next(
        col for col in train_df.columns if col.endswith(f"__{target}") or col == target
    )

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_val, y_val = test_df.drop(columns=[target_col]), test_df[target_col]

    start_time = datetime.now()
    model = GradientBoostingClassifier(**params).fit(X_train, y_train)
    end_time = datetime.now()

    # Save model locally
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)

    # Save model to Modal Volume
    model_checksum = save_artifact_to_modal(
        artifact=model,
        artifact_path=MODAL_MODEL_PATH,
    )

    # Log metadata
    log_metadata(
        metadata={
            "model_path": MODAL_MODEL_PATH,
            "model_checksum": model_checksum,
            "training_params": params,
            "val_accuracy": model.score(X_val, y_val),
            "training_start_time": start_time.isoformat(),
            "training_duration_seconds": (end_time - start_time).total_seconds(),
        }
    )

    # Log metadata about the model's purpose and limitations
    model_card_info = {
        "model_type": "GradientBoostingClassifier",
        "purpose": "Credit scoring prediction",
        "intended_use": "Evaluating loan applications",
        "limitations": "Model trained on historical data and may reflect historical biases",
        "performance_metrics": {"val_accuracy": model.score(X_val, y_val)},
    }

    # log metadata to model
    log_metadata(
        metadata={
            "model_card": model_card_info,
        },
        infer_model=True,
    )

    return model
