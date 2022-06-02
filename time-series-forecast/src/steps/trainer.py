from zenml.steps import step
from zenml.repository import Repository
import numpy as np
from sklearn.ensemble import RandomForestRegressor

step_operator = Repository().active_stack.step_operator

@step(custom_step_operator=step_operator.name)
def trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model
