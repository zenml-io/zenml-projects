from constants import SECRET_NAME
from zenml.client import Client


def get_openai_api_key():
    api_key = Client().get_secret(SECRET_NAME).secret_values["openai_api_key"]

    return api_key
