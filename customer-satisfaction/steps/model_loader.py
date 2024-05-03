from sklearn.base import RegressorMixin
from zenml import Model, step


@step
def model_loader(model_name: str) -> RegressorMixin:
    """Implements a simple model loader that loads the current production model.

    Args:
        model_name: Name of the Model to load
    """
    model = Model(name=model_name, version="production")
    model_artifact: RegressorMixin = model.load_artifact("sklearn_regressor")
    return model_artifact
