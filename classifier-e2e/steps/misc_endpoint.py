from typing_extensions import Annotated


from zenml import step
from sagemaker.predictor import Predictor
import pandas as pd


@step
def predict_on_endpoint(
    endpoint_name: str, dataset: pd.DataFrame
) -> Annotated[pd.Series, "real_time_predictions"]:
    predictor = Predictor(endpoint_name=endpoint_name)
    predictions = predictor.predict(
        data=dataset.to_csv(header=False, index=False),
        initial_args={"ContentType": "text/csv"},
    )
    return pd.Series(
        [float(l) for l in predictions.decode().split("\n") if l],
        name="predictions",
    )


@step
def shutdown_endpoint(endpoint_name: str):
    predictor = Predictor(endpoint_name=endpoint_name)
    predictor.delete_endpoint()
