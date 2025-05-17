import logging
from typing import Annotated, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from zenml import step

# Set up logger
logger = logging.getLogger(__name__)


@step
def train_xgb_model_with_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    feature_selection_threshold: str = "median",  # Can also be specific value
) -> Tuple[
    Annotated[xgb.XGBClassifier, "model"],
    Annotated[SelectFromModel, "feature_selector"],
]:
    """Trains an XGBoost classifier with built-in feature selection.

    Args:
        X_train: Training features.
        y_train: Training target.
        n_estimators: Number of trees in the model.
        max_depth: Maximum depth of each tree.
        learning_rate: Learning rate for the model.
        feature_selection_threshold: Threshold strategy or value for feature selection.

    Returns:
        Tuple containing the trained model and feature selector.
    """
    # Check for class imbalance
    class_counts = y_train.value_counts()
    class_ratio = min(class_counts) / max(class_counts)

    # Set scale_pos_weight for imbalanced dataset
    scale_pos_weight = 1.0
    if (
        class_ratio < 0.3
    ):  # If minority class is less than 30% of majority class
        # XGBoost recommends using the ratio of negative to positive instances
        scale_pos_weight = class_counts[0] / class_counts[1]

    # Initialize and train XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,  # Avoid future warning
        eval_metric="logloss",  # Specify eval metric to avoid warning
        random_state=42,
    )

    # Train the model on the original features
    model.fit(X_train, y_train)

    # Use the model for feature selection
    feature_selector = SelectFromModel(
        model, threshold=feature_selection_threshold, prefit=True
    )

    # Transform the training data
    X_train_selected = feature_selector.transform(X_train)

    # Retrain the model on selected features
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_selected, y_train)

    # Log feature selection info
    logger.info(f"Original number of features: {X_train.shape[1]}")
    logger.info(f"Number of features selected: {X_train_selected.shape[1]}")

    return model, feature_selector
