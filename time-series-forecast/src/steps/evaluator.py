from zenml.steps import step, Output
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

@step
def evaluator(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: RandomForestRegressor,
) -> float:
    
    y_pred = model.predict(X_test)
    score = r2_score(y_test,y_pred)
    print(f'R2 score: {score}')

    return score
