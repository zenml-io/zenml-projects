import logging
from typing import Tuple, Annotated

import mlflow
import numpy as np
import pandas as pd
from model.evaluator import Evaluator
from sklearn.base import RegressorMixin
from zenml.client import Client
from zenml import step, get_step_context, log_artifact_metadata

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: RegressorMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"]
]:
    """Evaluates the Model on the Test Dataset and returns the metrics.

    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series

    Returns:
        r2_score: float
        rmse: float
    """
    try:
        prediction = model.predict(x_test)
        evaluator = Evaluator()
        r2_score = evaluator.r2_score(y_test, prediction)
        mse = evaluator.mean_squared_error(y_test, prediction)
        rmse = np.sqrt(mse)

        # Log to MLFlow
        mlflow.log_metric("r2_score", r2_score)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # Also add the metrics to the Model within the ZenML Model Control Plane
        artifact = get_step_context().model_version.get_artifact("sklearn_regressor")

        log_artifact_metadata(
            metadata={
                "metrics": {
                    "r2_score": float(r2_score),
                    "mse": float(mse),
                    "rmse": float(rmse)
                }
            },
            artifact_name=artifact.name,
            artifact_version=artifact.version,
        )
        return mse, rmse
    except Exception as e:
        logging.error(e)
        raise e
