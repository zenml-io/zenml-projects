import pandas as pd
import os
import requests
from zenml import step

# Download the zip file
import zipfile
from io import BytesIO
import logging

# Set up logger
logger = logging.getLogger(__name__)


@step
def load_data(
    data_path: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
) -> pd.DataFrame:
    """Loads data from a CSV file or downloads it if a URL is provided.

    Args:
        data_path: Path to the CSV file or URL to download from.

    Returns:
        Pandas DataFrame loaded from the CSV.
    """
    # Case 1: Input is a URL - download the data
    if data_path.startswith(("http://", "https://")):
        logger.info(f"Downloading data from URL: {data_path}")
        try:
            response = requests.get(data_path)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Extract the zip file
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                # Find the CSV file (there might be multiple CSVs, we want the one with all data)
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    raise Exception("No CSV files found in the zip archive")

                bank_csv = next(
                    (f for f in csv_files if "full" in f), csv_files[0]
                )

                # Extract the CSV file
                with z.open(bank_csv) as f:
                    # Read the CSV directly from the zip
                    data = pd.read_csv(f, sep=";")
                    return data

        except Exception as e:
            raise Exception(
                f"Failed to download dataset from {data_path}: {str(e)}"
            )

    # Case 2: Input is a local file that exists
    elif os.path.exists(data_path):
        logger.info(f"Loading data from local file: {data_path}")
        try:
            data = pd.read_csv(data_path, sep=";")
            return data
        except Exception as e:
            raise Exception(f"Error loading {data_path}: {str(e)}")

    # Case 3: Neither a valid URL nor an existing file
    else:
        raise FileNotFoundError(
            f"{data_path} is not a valid URL or existing file path"
        )
