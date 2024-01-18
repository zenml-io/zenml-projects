# Write a iris zenml pipeline

from zenml  import pipeline
from zenml.steps import step

@step
def importer() -> pd.DataFrame:
    """Load the iris dataset as a Pandas DataFrame."""
    iris = load_iris(as_frame=True)
    return iris.frame

@step
def normalizer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the data by subtracting the mean and dividing by the standard deviation."""
    X_train_norm =  X_train - X_train.mean(axis=0)
    X_train_norm = X_train_norm / X_train.std(axis=0)
    X_test_norm = X_test - X_train.mean(axis=0)
    return x_train, x_test 

@pipeline 

def my_pipeline(
    importer,
    normalizer,
):
    X_train, X_test, y_train, y_test = importer()
    X_train_norm, X_test_norm = normalizer(X_train, X_test)

