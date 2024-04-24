from typing import Annotated

from pipelines.utils import get_data_for_test
from zenml import step


@step(enable_cache=False)
def dynamic_importer() -> Annotated[str, "batch_data"]:
    """Downloads the latest data from a mock API."""
    batch_data = get_data_for_test()
    return batch_data
