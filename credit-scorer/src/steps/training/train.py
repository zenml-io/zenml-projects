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
from typing import Annotated, Dict, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from zenml import ArtifactConfig, log_metadata, step
from zenml.enums import ArtifactType
from zenml.logger import get_logger

from src.constants import (
    MODEL_NAME,
    TEST_DATASET_NAME,
    TRAIN_DATASET_NAME,
)
from src.utils import save_artifact_to_modal

logger = get_logger(__name__)


@step()
def train_model(
    train_df: Annotated[pd.DataFrame, TRAIN_DATASET_NAME],
    test_df: Annotated[pd.DataFrame, TEST_DATASET_NAME],
    target: str = "target",
    hyperparameters: Optional[Dict] = None,
    model_path: str = "models/lgbm_model.pkl",
) -> Annotated[
    lgb.LGBMClassifier,
    ArtifactConfig(name=MODEL_NAME, artifact_type=ArtifactType.MODEL),
]:
    """Train LightGBM with balanced objective, early stopping, and threshold tuning."""
    # Identify target column
    target_col = next(
        col for col in train_df.columns if col.endswith(f"__{target}") or col == target
    )

    # Create copies and clean feature names for LightGBM
    X_train = train_df.drop(columns=[target_col]).copy()
    y_train = train_df[target_col].copy()
    X_val = test_df.drop(columns=[target_col]).copy()
    y_val = test_df[target_col].copy()

    # Clean feature names (replace special characters)
    feature_name_map = {}
    cleaned_columns = []

    for i, col in enumerate(X_train.columns):
        # Create a clean name: replace special characters with underscore
        clean_name = f"feature_{i}"
        feature_name_map[clean_name] = col
        cleaned_columns.append(clean_name)

    # Rename columns with clean names
    X_train.columns = cleaned_columns
    X_val.columns = cleaned_columns

    # Log class imbalance
    class_counts = y_train.value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()
    logger.info(f"Class distribution: {class_counts}")
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.4f}")

    # Calculate scale factor for imbalanced data handling
    neg_count = class_counts.get(0, 0)
    pos_count = class_counts.get(1, 0)
    scale_pos_weight = float(neg_count) / max(pos_count, 1)

    # Base hyperparameters (merge any overrides)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "n_estimators": 200,
        "scale_pos_weight": scale_pos_weight,  # Handle class imbalance
        "random_state": 42,
        "verbosity": -1,  # Silence output
    }
    if hyperparameters:
        params.update(hyperparameters)

    logger.info(f"Training LGBMClassifier with params: {params}")

    # Train with early stopping, silence output
    start_time = datetime.now()
    model = lgb.LGBMClassifier(**params)

    # Use callbacks for early stopping
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress output
        ],
    )
    end_time = datetime.now()

    # Threshold tuning on validation set
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    f1s = [f1_score(y_val, probs >= t) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])
    logger.info(f"Optimal threshold on validation: {best_t:.2f} → F1 = {best_f1:.4f}")

    # Compute validation metrics at optimal threshold
    y_pred = (probs >= best_t).astype(int)
    val_results = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1_score": float(f1_score(y_val, y_pred)),
        "auc": float(roc_auc_score(y_val, probs)),
        "optimal_threshold": best_t,
    }
    logger.info(f"Validation metrics: {val_results}")

    # Log top‑10 feature importances (safely with original names)
    importance = pd.Series(model.feature_importances_, index=cleaned_columns)
    # Map back to original feature names for logging
    importance.index = [feature_name_map.get(feat, feat) for feat in importance.index]
    top_features = list(importance.sort_values(ascending=False).head(10).items())
    logger.info(f"Top 10 features: {top_features}")

    # Save feature name mapping with the model
    model_metadata = {
        "feature_name_map": feature_name_map,
        "original_column_order": list(train_df.drop(columns=[target_col]).columns),
        "cleaned_column_order": cleaned_columns,
    }

    # Save model locally & to Modal volume
    joblib.dump((model, model_metadata), model_path)
    model_checksum = save_artifact_to_modal(
        artifact=(model, model_metadata),
        artifact_path=model_path,
    )

    # Log metadata to ZenML
    log_metadata(
        metadata={
            "model_type": "LGBMClassifier",
            "training_params": params,
            "model_checksum": model_checksum,
            "validation_metrics": val_results,
            "best_iteration": int(model.best_iteration_)
            if hasattr(model, "best_iteration_")
            else None,
            "feature_importance": top_features,
            "training_start_time": start_time.isoformat(),
            "training_duration_seconds": (end_time - start_time).total_seconds(),
            "feature_name_mapping": feature_name_map,  # Include mapping for reference
        }
    )

    # Create and log model card info
    model_card = {
        "model_type": "LGBMClassifier",
        "purpose": "Credit scoring prediction",
        "intended_use": "Evaluating loan applications",
        "limitations": "Model trained on historical data may reflect historical biases",
        "performance_metrics": val_results,
    }
    log_metadata(
        metadata={"model_card": model_card},
        infer_model=True,
    )

    return model
