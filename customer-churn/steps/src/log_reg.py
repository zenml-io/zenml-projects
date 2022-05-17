import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from zenml.logger import get_logger

logger = get_logger(__name__)
from rich import print as rprint


class LogisticRegression:
    def __init__(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        assumptions_test: bool = True,
    ) -> None:
        """Initialize the Logistic Regression model.
        x_train: training data
        x_test: testing data
        y_train: training labels
        y_test: testing labels
        assumptions_test: boolean, if True, test assumptions
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.assumptions_test = assumptions_test

    def main(self) -> None:
        """Run the Logistic Regression model."""
        try:
            logger.info("Started Training Logistic Regression model.")
            self.fit()
            if self.assumptions_test:
                self.assumption_appropriate_outcome_type()
            logger.info("Finished Training Logistic Regression model.")
        except Exception as e:
            logger.error(f"Logistic Regression model failed: {e}")

    def fit(self) -> None:
        """Fit the Logistic Regression model."""
        try:
            log_reg = sm.Logit(self.y_train, self.x_train)
            log_reg = log_reg.fit()
            rprint(log_reg.summary())
            return log_reg
        except Exception as e:
            logger.error(f"Logistic Regression model failed to fit: {e}")

    def assumption_appropriate_outcome_type(self) -> None:
        """Test assumptions for appropriate outcome type."""
        try:
            if self.y_train.nunique() == 2:
                logger.info("Assumption for appropriate outcome type is satisfied.")
            else:
                logger.error("Assumption for appropriate outcome type is not satisfied.")
        except Exception as e:
            logger.error(f"Assumptions test failed: {e}")
