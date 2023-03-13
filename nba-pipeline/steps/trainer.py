import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from zenml.client import Client
from zenml.steps import BaseParameters, step

experiment_tracker = Client().active_stack.experiment_tracker


class RandomForestTrainerConfig(BaseParameters):
    """Config class for the sklearn trainer.

    Attributes:
        max_depth: Maximum depth of the tree during training.
        target_col: Target column name.
    """

    max_depth: int = 10000
    target_col: str = "FG3M"


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def random_forest_trainer(
    train_df_x: pd.DataFrame,
    train_df_y: pd.DataFrame,
    eval_df_x: pd.DataFrame,
    eval_df_y: pd.DataFrame,
    config: RandomForestTrainerConfig,
) -> RegressorMixin:
    """Trains a random forest.

    Args:
        train_df_x: DataFrame with training feature data.
        train_df_y: DataFrame with training label data.
        eval_df_x: DataFrame with eval feature data.
        eval_df_y: DataFrame with eval label data.
        config: Runtime parameters of the training process.

    Returns:
        RegressorMixin: [description]
    """
    mlflow.sklearn.autolog()
    clf = RandomForestRegressor(max_depth=config.max_depth)
    clf.fit(train_df_x, np.squeeze(train_df_y.values.T))
    eval_score = clf.score(eval_df_x, np.squeeze(eval_df_y.values.T))
    print(f"Eval score is: {eval_score}")
    return clf
