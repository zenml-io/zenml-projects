import os
from typing import Optional

from zenml.client import Client


def get_huggingface_access_token() -> Optional[str]:
    try:
        return Client().get_secret("huggingface_credentials").secret_values["token"]
    except KeyError:
        return os.getenv("HF_TOKEN")
