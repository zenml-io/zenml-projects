import pandas as pd
from sagemaker.predictor import Predictor
from typing_extensions import Annotated
from zenml import step


@step
def predict_on_endpoint(
    predictor: Predictor, dataset: pd.DataFrame
) -> Annotated[pd.Series, "real_time_predictions"]:
    predictions = predictor.predict(
        data=dataset.to_csv(header=False, index=False),
        initial_args={"ContentType": "text/csv"},
    )
    return pd.Series(
        [float(l) for l in predictions.decode().split("\n") if l],
        name="predictions",
    )


@step
def shutdown_endpoint(predictor: Predictor):
    predictor.delete_endpoint()
