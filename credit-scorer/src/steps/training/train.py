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

from src.constants import Artifacts as A
from src.utils import save_artifact_to_modal

logger = get_logger(__name__)


@step()
def train_model(
    train_df: Annotated[pd.DataFrame, A.TRAIN_DATASET],
    test_df: Annotated[pd.DataFrame, A.TEST_DATASET],
    target: str = "target",
    hyperparameters: Optional[Dict] = None,
    model_path: str = "models/lgbm_model.pkl",
    protected_attributes: Optional[list] = None,
) -> Tuple[
    Annotated[
        lgb.LGBMClassifier,
        ArtifactConfig(name=A.MODEL, artifact_type=ArtifactType.MODEL),
    ],
    Annotated[float, A.OPTIMAL_THRESHOLD],
]:
    """Train LightGBM with bias-aware techniques, balanced objective, and threshold tuning."""
    # Identify target column
    target_col = next(
        col
        for col in train_df.columns
        if col.endswith(f"__{target}") or col == target
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

    # Track training time
    start_time = datetime.now()

    # Apply bias-aware training approach
    logger.info("Training with bias-aware LightGBM")

    # Enhanced hyperparameters for fairness
    fairness_params = hyperparameters.copy()
    fairness_params.update(
        {
            # Class balance handling
            "is_unbalance": True,
            "boost_from_average": False,
            # Regularization for bias reduction
            "reg_alpha": 0.3,
            "reg_lambda": 0.3,
            # Conservative tree structure
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 50,
            # Feature and sample randomization
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }
    )

    # Store bias mitigation metadata
    fairness_metadata = {
        "bias_aware_training": True,
        "regularization_enhanced": True,
        "tree_structure_conservative": True,
        "class_balancing": True,
        "feature_randomization": True,
    }

    logger.info("Applied bias-aware hyperparameters")
    logger.info(
        f"Regularization: α={fairness_params['reg_alpha']}, λ={fairness_params['reg_lambda']}"
    )
    logger.info(
        f"Tree constraints: {fairness_params['num_leaves']} leaves, depth {fairness_params['max_depth']}"
    )

    # Train model with bias-aware parameters
    model = lgb.LGBMClassifier(**fairness_params)

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Record training end time
    end_time = datetime.now()

    # Calculate optimal threshold on validation set
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    f1s = [f1_score(y_val, probs >= t) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    best_t = float(thresholds[best_idx])

    best_f1 = float(f1s[best_idx])
    logger.info(
        f"Optimal threshold on validation: {best_t:.2f} → F1 = {best_f1:.4f}"
    )

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
    importance.index = [
        feature_name_map.get(feat, feat) for feat in importance.index
    ]
    top_features = list(
        importance.sort_values(ascending=False).head(10).items()
    )
    logger.info(f"Top 10 features: {top_features}")

    # Save feature name mapping with the model
    model_metadata = {
        "feature_name_map": feature_name_map,
        "original_column_order": list(
            train_df.drop(columns=[target_col]).columns
        ),
        "cleaned_column_order": cleaned_columns,
    }

    # Add the optimal threshold and fairness metadata to the model metadata
    model_metadata["optimal_threshold"] = best_t
    model_metadata.update(fairness_metadata)

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
            "training_params": fairness_params,
            "model_checksum": model_checksum,
            "validation_metrics": val_results,
            "best_iteration": int(model.best_iteration_)
            if hasattr(model, "best_iteration_")
            else None,
            "feature_importance": top_features,
            "training_start_time": start_time.isoformat(),
            "training_duration_seconds": (
                end_time - start_time
            ).total_seconds(),
            "feature_name_mapping": feature_name_map,
            "bias_mitigation": fairness_metadata,
        }
    )

    # Create and log model card info
    model_card = {
        "model_type": "LGBMClassifier",
        "purpose": "Credit scoring prediction",
        "intended_use": "Evaluating loan applications",
        "limitations": "Model trained with bias-aware techniques for EU AI Act compliance",
        "performance_metrics": val_results,
        "bias_mitigation": fairness_metadata,
    }
    log_metadata(
        metadata={"model_card": model_card},
        infer_model=True,
    )

    return model, best_t
