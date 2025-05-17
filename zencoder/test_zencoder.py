# Write a zenml pipeline that loads sklearn iris dataset and builds a sklearn classifier

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zenml import pipeline, step


@step
def importer() -> pd.DataFrame:
    """Load the iris dataset."""
    df = load_iris(as_frame=True)["data"]
    return df


@step
def trainer(df: pd.DataFrame) -> Any:
    """Train a model on the dataset."""
    X_train, X_test, y_train, y_test = train_test_split(
        df.to_numpy()[:, :2],
        df.to_numpy()[:, 2],
        test_size=0.2,
        random_state=42,
    )
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    return model


@pipeline
def sklearn_pipeline():
    df = importer()
    trainer(df)
