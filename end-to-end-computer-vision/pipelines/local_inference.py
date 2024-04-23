import fiftyone as fo
from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger

from utils.constants import PREDICTIONS_DATASET_ARTIFACT_NAME

logger = get_logger(__name__)


@step
def create_fiftyone_dataset():
    artifact = Client().get_artifact_version(
        name_id_or_prefix=PREDICTIONS_DATASET_ARTIFACT_NAME
    )
    dataset_json = artifact.load()
    dataset = fo.Dataset.from_json(dataset_json, persistent=False)
    fo.launch_app(dataset)


@pipeline(enable_cache=False)
def fifty_one():
    create_fiftyone_dataset()
