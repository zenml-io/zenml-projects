from constants import SECRET_NAME
from zenml.client import Client


def get_hf_token() -> str:
    api_key = Client().get_secret(SECRET_NAME).secret_values["hf_token"]

    return api_key
