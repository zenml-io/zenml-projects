"""
Fine-Tune StarCoder on code/text dataset

Based off Sayak Paul (https://github.com/sayakpaul) and Sourab Mangrulkar (https://github.com/pacman100) codebase: https://github.com/pacman100/DHS-LLM-Workshop/tree/main/
All credit to them for their amazing work!
"""

import os
import pandas as pd
from nbformat import reads, NO_CONVERT
from tqdm import tqdm
from datasets import Dataset
from typing import Dict
from huggingface_hub import HfApi
from zenml import step
from zenml.client import Client

SERIALIZE_IN_CHUNKS = 10000
FEATHER_FORMAT = "ftr"

# Block the following formats.
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = [
    "key",
    "PDF",
    "pdf",
    "docx",
    "xlsx",
    "pptx",
]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy",
    "index",
    "inv",
    "index",
    "DS_Store",
    "rdb",
    "pack",
    "idx",
    "glb",
    "gltf",
    "len",
    "otf",
    "unitypackage",
    "ttf",
    "xz",
    "pcm",
    "opus",
]
ANTI_FOMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + OTHERS)


def upload_to_hub(df: pd.DataFrame, dataset_id: str) -> str:
    """Uploads the DataFrame to the Hugging Face Hub."""
    secret = Client().get_secret("huggingface_creds")
    token = secret.secret_values["token"]
    api = HfApi(token=token)

    repo_id = api.create_repo(repo_id=dataset_id, exist_ok=True, repo_type="dataset").repo_id

    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(dataset_id, token=token)
    return repo_id

def filter_code_cell(cell) -> bool:
    """Filters a code cell w.r.t shell commands, etc."""
    only_shell = cell["source"].startswith("!")
    only_magic = "%%capture" in cell["source"]
    if only_shell or only_magic:
        return False
    else:
        return True

def process_file(directory_name: str, file_path: str) -> Dict[str, str]:
    """Processes a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            if file_path.endswith("ipynb"):
                # Code courtesy: Chansung Park and Sayak Paul.
                code_cell_str = ""
                notebook = reads(content, NO_CONVERT)

                code_cells = [
                    c
                    for c in notebook["cells"]
                    if c["cell_type"] == "code"
                    if filter_code_cell(c)
                ]

                for cell in code_cells:
                    code_cell_str += cell["source"]
                content = code_cell_str
    except Exception:
        content = ""

    return {
        "repo_id": directory_name,
        "file_path": file_path,
        "content": content,
    }


def read_repository_files(directory) -> pd.DataFrame:
    """Reads the files from the locally cloned repositories."""
    file_paths = []
    df = pd.DataFrame(columns=["repo_id", "file_path", "content"])

    # Recursively find all files within the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not file_path.endswith(ANTI_FOMATS) and all(
                k not in file_path for k in [".git", "__pycache__", "xcodeproj"]
            ):
                file_paths.append((os.path.dirname(root), file_path))

    # Process files sequentially.
    print(f"Total file paths: {len(file_paths)}.")
    print("Reading file contents...")

    for i, (directory_name, file_path) in enumerate(tqdm(file_paths)):
        file_content = process_file(directory_name, file_path)

        if file_content["content"] != "":
            temp_df = pd.DataFrame.from_dict([file_content])
            df = pd.concat([df, temp_df])

    return df


@step(enable_cache=False)
def prepare_dataset(mirror_directory: str, dataset_id: str = "zenml-codegen-v1"):
    df = read_repository_files(mirror_directory)
    repo_id = upload_to_hub(df, dataset_id)
    return repo_id
