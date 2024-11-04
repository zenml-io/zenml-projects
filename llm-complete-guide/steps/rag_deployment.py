import os
import webbrowser

from huggingface_hub import HfApi

from utils.hf_utils import get_hf_token
from utils.llm_utils import process_input_with_retrieval
from zenml import step
from zenml.client import Client
from zenml.integrations.registry import integration_registry

secret = Client().get_secret("llm-complete")

ZENML_API_TOKEN = secret.secret_values["zenml_api_token"]
ZENML_STORE_URL = secret.secret_values["zenml_store_url"]
SPACE_USERNAME = os.environ.get("ZENML_HF_USERNAME", "zenml")
SPACE_NAME = os.environ.get("ZENML_HF_SPACE_NAME", "llm-complete-guide-rag")

hf_repo_id = f"{SPACE_USERNAME}/{SPACE_NAME}"
gcp_reqs = integration_registry.select_integration_requirements("gcp")

hf_repo_requirements = f"""
zenml>=0.68.1
ratelimit
pgvector
psycopg2-binary
beautifulsoup4
pandas
openai
numpy
sentence-transformers>=3
transformers
litellm
tiktoken
matplotlib
pyarrow
rerankers[flashrank]
datasets
torch
huggingface-hub
{chr(10).join(gcp_reqs)}
"""


def predict(message, history):
    return process_input_with_retrieval(
        input=message,
        n_items_retrieved=20,
        use_reranking=True,
    )


def upload_files_to_repo(
    api, repo_id: str, files_mapping: dict, token: str
):
    """Upload multiple files to a Hugging Face repository

    Args:
        api: Hugging Face API client
        repo_id: Target repository ID
        files_mapping: Dict mapping local files to repo destinations
        token: HF API token
    """
    for local_path, repo_path in files_mapping.items():
        content = (
            local_path.encode()
            if isinstance(local_path, str) and not os.path.exists(local_path)
            else local_path
        )
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )


@step(enable_cache=False)
def gradio_rag_deployment() -> None:
    """Launches a Gradio chat interface with the slow echo demo.

    Starts a web server with a chat interface that echoes back user messages.
    The server runs indefinitely until manually stopped.
    """
    api = HfApi()
    api.create_repo(
        repo_id=hf_repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=True,
        exist_ok=True,
        token=get_hf_token(),
    )
    api.add_space_secret(
        repo_id=hf_repo_id,
        key="ZENML_STORE_API_KEY",
        value=ZENML_API_TOKEN,
    )
    api.add_space_secret(
        repo_id=hf_repo_id,
        key="ZENML_STORE_URL",
        value=ZENML_STORE_URL,
    )

    files_to_upload = {
        "deployment_hf.py": "app.py",
        "utils/llm_utils.py": "utils/llm_utils.py",
        "utils/openai_utils.py": "utils/openai_utils.py",
        "utils/__init__.py": "utils/__init__.py",
        "constants.py": "constants.py",
        "structures.py": "structures.py",
        hf_repo_requirements: "requirements.txt",
    }

    upload_files_to_repo(api, hf_repo_id, files_to_upload, get_hf_token())

    webbrowser.open(f"https://huggingface.co/spaces/{hf_repo_id}")
