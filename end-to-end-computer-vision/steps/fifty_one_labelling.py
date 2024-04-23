import fiftyone as fo
from zenml import step
from zenml.client import Client

from utils.constants import PREDICTIONS_DATASET_ARTIFACT_NAME


@step
def create_fiftyone_dataset():
    artifact = Client().get_artifact_version(
        name_id_or_prefix=PREDICTIONS_DATASET_ARTIFACT_NAME
    )
    dataset_json = artifact.load()
    dataset = fo.Dataset.from_json(dataset_json, persistent=False)
    fo.launch_app(dataset)
