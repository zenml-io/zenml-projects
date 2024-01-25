import numpy as np
import pandas as pd
from zenml import step
from zenml.logger import get_logger
from typing_extensions import Annotated

logger = get_logger(__name__)


@step
def cpi_data_loader_step(
    data_stream: str,
) -> Annotated[pd.DataFrame, "month_data"]:
    """Loads the air passenger data."""
    if data_stream == "acme":
        df = pd.read_csv(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        )
    else:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        )

    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month")

    # Assume the data is in 5-minute intervals for the sake of the example
    df = (
        df.resample("5T").ffill().iloc[:8928]
    )  # Select one month of data (8928 5-minute intervals)

    return df
