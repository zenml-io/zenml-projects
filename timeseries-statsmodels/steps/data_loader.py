import numpy as np
import pandas as pd
from zenml import step
from pandas_datareader.data import DataReader
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def cpi_data_loader_step(
    start_date: str = "1971-01", end_date: str = "2023-12"
) -> np.ndarray:
    """Loads the CPI data from FRED and preprocesses it."""
    # Load CPI data from FRED using pandas_datareader
    logger.info("Data loading...")
    cpi = DataReader("CPIAUCSL", "fred", start=start_date, end=end_date)
    cpi.index = pd.DatetimeIndex(cpi.index, freq="MS")

    # Define the inflation series for analysis
    inf = np.log(cpi).resample("QS").mean().diff()[1:] * 400
    inf = inf.dropna()
    inf_array = inf.values.flatten()  # Convert to 1D numpy array
    return inf_array
