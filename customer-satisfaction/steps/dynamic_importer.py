from typing import Annotated

from zenml import step

from materializer.custom_materializer import cs_materializer
from pipelines.utils import get_data_for_test


@step(enable_cache=False, output_materializers=cs_materializer)
def dynamic_importer() -> Annotated[str, "data"]:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data
