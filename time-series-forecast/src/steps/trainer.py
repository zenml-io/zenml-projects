import numpy as np
from sklearn.ensemble import RandomForestRegressor
from zenml.repository import Repository
from zenml.steps import step

step_operator = Repository().active_stack.step_operator


@step(step_operator=step_operator.name)
def trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    """Trains a random forest.

    Args:
        X_train: DataFrame with training feature data.
        y_train: DataFrame with training target data.

    Returns:
        RegressorMixin: [description]
    """

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model
