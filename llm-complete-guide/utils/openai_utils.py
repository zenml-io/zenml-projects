import os

from zenml.client import Client


def get_openai_api_key():
    secret_name = os.getenv("ZENML_OPENAI_SECRET_NAME")

    if not secret_name:
        raise RuntimeError(
            "Please make sure to set the environment variable: ZENML_OPENAI_SECRET_NAME to point at the secret that "
            "contains your openai api key."
        )

    api_key = (
        Client()
        .get_secret(secret_name)
        .secret_values["api_key"]
    )

    return api_key