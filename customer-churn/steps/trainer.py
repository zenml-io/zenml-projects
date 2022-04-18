import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import step

from .src.log_reg import LogisticRegression


@step
def log_reg_trainer(train: pd.DataFrame) -> LogisticRegression:
    """Train a logistic regression model."""
    x_train, x_test, y_train, y_test = train_test_split(
        train.drop("Churn", axis=1), train["Churn"], test_size=0.2
    )
    log_reg = LogisticRegression(x_train, x_test, y_train, y_test, assumptions_test=True)
    log_reg.main()
    return log_reg
