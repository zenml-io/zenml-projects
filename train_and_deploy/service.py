import bentoml
import numpy as np
from bentoml.validators import Shape
from typing_extensions import Annotated


@bentoml.service
class GitGuarden:
    """
    A simple service using a sklearn model
    """

    # Load in the class scope to declare the model as a dependency of the service
    iris_model = bentoml.models.get("gitguarden:latest")

    def __init__(self):
        """
        Initialize the service by loading the model from the model store
        """
        import joblib

        self.model = joblib.load(self.iris_model.path_of("saved_model.pkl"))

    @bentoml.api
    def predict(
        self,
        input_series: Annotated[np.ndarray, Shape((-1, 30))],
    ) -> np.ndarray:
        """
        Define API with preprocessing and model inference logic
        """
        return self.model.predict(input_series)