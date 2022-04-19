import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step

from .src.configs import StackEnsembleConfig
from .src.log_reg import LogisticRegression
from .src.stacking_models import StackedEnsembles


@step
def log_reg_trainer(train: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression model."""
    x_train, x_test, y_train, y_test = train_test_split(
        train.drop("Churn", axis=1), train["Churn"], test_size=0.2
    )
    log_reg = LogisticRegression(x_train, x_test, y_train, y_test, assumptions_test=True)
    log_reg.main()
    return log_reg


@step(enable_cache=True)
def stacking_level1_trainer(
    config: StackEnsembleConfig, train: pd.DataFrame, test: pd.DataFrame
) -> Output(stack_x_train=np.ndarray, stack_x_test=np.ndarray):
    """Train a stacked models model."""
    stk = StackedEnsembles(config, train, test)
    stack_x_train, stack_x_test = stk.stacking_model_builder()
    return stack_x_train, stack_x_test
