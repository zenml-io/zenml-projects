import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from zenml.logger import get_logger
from zenml.steps import step


@step
def evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: RandomForestRegressor,
) -> float:
    """Evaluate a random forest using R2 score.

    Args:
        X_test: DataFrame with eval feature data.
        y_test: DataFrame with eval target data.
        model: Trained Random Forest Regressor.

    Returns:
        float
    """
    logger = get_logger(__name__)
    try:
        y_pred = model.predict(X_test)
    except:
        logger.info("Error occurred when predicting on test data")

    score = r2_score(y_test, y_pred)
    logger.info(f"R2 score: {score}")

    return score
