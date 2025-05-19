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

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Optional

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from zenml import log_metadata, step


@step
def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str = "loan_approved",
    hyperparameters: Optional[Dict] = None,
) -> Annotated[str, "model_path"]:
    """Train a GradientBoosting model.

    Logs hyper-params & model checksum (Annex IV §2 b,g).

    Args:
        train_df: Training dataset.
        val_df: Validation dataset.
        target: Target column name.
        hyperparameters: Hyperparameters for the model.

    Returns:
        Path to the trained model.
    """
    params = hyperparameters or {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
    }

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]

    start_time = datetime.now()
    model = GradientBoostingClassifier(**params).fit(X_train, y_train)
    end_time = datetime.now()

    # Persist model
    model_path = "model.pkl"
    joblib.dump(model, model_path)

    # Checksum for integrity
    sha256 = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    log_metadata(
        metadata={
            "model_uri": model_path,
            "model_sha256": sha256,
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

    log_metadata(
        metadata={
            "model_card": model_card_info,
        }
    )

    return model_path  # downstream steps can load via joblib
