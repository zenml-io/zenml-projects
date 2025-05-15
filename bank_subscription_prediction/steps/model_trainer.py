import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from zenml import step
from typing import Annotated, Tuple

@step
def train_xgb_model_with_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 3,
    min_child_weight: int = 1,
    gamma: float = 0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    objective: str = 'binary:logistic',
    scale_pos_weight: float = 1, # Default to 1, adjust based on imbalance
    random_state: int = 42,
    feature_selection_threshold: str = "median", # Threshold for SelectFromModel
) -> Tuple[Annotated[xgb.XGBClassifier, "trained_model"], Annotated[SelectFromModel, "feature_selector"]]:
    """Trains an XGBoost classifier with feature selection.

    Args:
        X_train: Training features.
        y_train: Training target.
        learning_rate: Learning rate for XGBoost.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum depth of a tree.
        min_child_weight: Minimum sum of instance weight needed in a child.
        gamma: Minimum loss reduction required to make a further partition.
        subsample: Subsample ratio of the training instance.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        objective: Specify the learning task and the corresponding learning objective.
        scale_pos_weight: Control the balance of positive and negative weights.
        random_state: Random seed for reproducibility.
        feature_selection_threshold: Threshold for SelectFromModel (e.g., "median", "mean", float).

    Returns:
        A tuple containing the trained XGBoost model and the feature selector.
    """
    
    model = xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective=objective,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False # Suppress warning for newer XGBoost versions
    )

    # Feature Selection using SelectFromModel
    # The notebook fits SelectFromModel on the training data before training the final model
    # This is a common practice, though sometimes it's done within a CV loop.
    # Here, we select features based on the initial model fit on X_train.
    
    # Fit an initial model for feature selection
    selection_model = xgb.XGBClassifier(
        random_state=random_state, use_label_encoder=False
    )
    selection_model.fit(X_train, y_train)
    
    fs_selector = SelectFromModel(selection_model, threshold=feature_selection_threshold, prefit=True)
    
    # Transform X_train to selected features
    X_train_selected = fs_selector.transform(X_train)
    
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of features selected: {X_train_selected.shape[1]}")

    # Train the final model on the selected features
    model.fit(X_train_selected, y_train)
    
    return model, fs_selector 