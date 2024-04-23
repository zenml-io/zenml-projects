from zenml import pipeline
from zenml.logger import get_logger

from steps.fifty_one_labelling import create_fiftyone_dataset

logger = get_logger(__name__)


@pipeline
def fifty_one():
    create_fiftyone_dataset()
