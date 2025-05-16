import pandas as pd
import os
import requests
from zenml import step


@step
def load_data(csv_file_path: str = "data/bank.csv") -> pd.DataFrame:
    """Loads data from a CSV file or downloads it if not available.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        Pandas DataFrame loaded from the CSV.
    """
    # Check if file exists locally
    if not os.path.exists(csv_file_path):
        print(
            f"File {csv_file_path} not found. Downloading from UCI ML Repository..."
        )

        # URL for the bank marketing dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

        try:
            # Download the zip file
            import zipfile
            from io import BytesIO

            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Extract the zip file
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                # Find the CSV file (there might be multiple CSVs, we want the one with all data)
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                bank_csv = next(
                    (f for f in csv_files if "full" in f), csv_files[0]
                )

                # Extract the CSV file
                with z.open(bank_csv) as f:
                    # Read the CSV directly from the zip
                    data_raw_all = pd.read_csv(f, sep=";")

                    # Save locally for future use
                    data_raw_all.to_csv(csv_file_path, sep=";", index=False)

                    print(f"Dataset downloaded and saved to {csv_file_path}")
                    return data_raw_all

        except Exception as e:
            raise Exception(f"Failed to download dataset: {str(e)}")

    try:
        # If we reach here, the file exists locally
        data_raw_all = pd.read_csv(csv_file_path, header=0, sep=";")
        return data_raw_all
    except Exception as e:
        raise Exception(f"Error loading {csv_file_path}: {str(e)}")
