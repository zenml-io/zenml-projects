"""
Fine-Tune StarCoder on code/text dataset

Based off Sayak Paul (https://github.com/sayakpaul) and Sourab Mangrulkar (https://github.com/pacman100) codebase: https://github.com/pacman100/DHS-LLM-Workshop/tree/main/
All credit to them for their amazing work!
"""
import glob

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from zenml import step
from zenml.client import Client

FEATHER_FORMAT = "*.ftr"


@step
def push_to_hub(repo_id: str, dataset_id: str):
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        repo_id (str): The name of the repo to create/use on huggingface.
        dataset_id (str): The name of the dataset to create/use on huggingface.
    """
    secret = Client().get_secret("huggingface_creds")
    token = secret.secret_values["token"]
    api = HfApi(token=token)

    folder_path = api.snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"*.{FEATHER_FORMAT}",
        repo_type="dataset",
    )
    feather_files = glob.glob(f"{folder_path}/raw_csvs/*.{FEATHER_FORMAT}")
    print(folder_path, len(feather_files))

    all_dfs = []

    for feather_file in tqdm(feather_files):
        df = pd.read_feather(feather_file)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs)
    print(f"Final DF prepared containing {len(final_df)} rows.")

    dataset = Dataset.from_pandas(final_df)
    dataset.push_to_hub(dataset_id)
    print("Dataset pushed to the Hugging Face Hub.")
