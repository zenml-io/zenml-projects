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
                self.linearity_assumption_of_logodds()
            logger.info("Finished Training Logistic Regression model.")
        except Exception as e:
            logger.error(f"Logistic Regression model failed: {e}")

    def fit(self) -> None:
        """Fit the Logistic Regression model."""
        try:
            log_reg = sm.Logit(self.y_train, self.x_train)
            log_reg = self.model.fit()
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

    def linearity_assumption_of_logodds(self) -> None:
        """Test assumptions for linearity of logodds."""
        try:
            continuous_var = ["SeniorCitizen"]
            # Add logit transform interaction terms (natural log) for continuous variables e.g.. Age * Log(Age)
            for var in continuous_var:
                self.x_train[f"{var}:Log_{var}"] = self.x_train[var].apply(lambda x: x * np.log(x))
            cols_to_keep = continuous_var + self.x_train.columns.tolist()[-len(continuous_var) :]
            X_lt = self.x_train[cols_to_keep]
            X_lt_constant = sm.add_constant(X_lt, prepend=False)
            logit_results = GLM(
                self.y_train.astype(float), X_lt_constant.astype(float), family=families.Binomial()
            ).fit()
            rprint(logit_results.summary())
        except Exception as e:
            logger.error(f"Assumptions test failed: {e}")


if __name__ == "__main__":
    data = pd.read_csv(
        "/home/ayushsingh/Documents/zenfiles/customer-churn/data/customer-churn-data.csv"
    )
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop("Churn", axis=1), data["Churn"], test_size=0.2, random_state=0
    )
    log_reg = LogisticRegression(x_train, x_test, y_train, y_test)
    log_reg.main()
