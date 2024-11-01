import os
import webbrowser

from huggingface_hub import (
    HfApi,
)
from utils.llm_utils import process_input_with_retrieval
from zenml import step
from zenml.integrations.registry import integration_registry

repo_id = "strickvl/my-test-space"
gcp_reqs = integration_registry.select_integration_requirements("gcp")


repo_requirements = f"""
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

HF_TOKEN = os.getenv("HF_TOKEN")


def predict(message, history):
    return process_input_with_retrieval(
        input=message,
        n_items_retrieved=20,
        use_reranking=True,
    )


@step
def gradio_rag_deployment() -> None:
    """Launches a Gradio chat interface with the slow echo demo.

    Starts a web server with a chat interface that echoes back user messages.
    The server runs indefinitely until manually stopped.
    """
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=True,
        exist_ok=True,
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="deployment_hf.py",
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="utils/llm_utils.py",
        path_in_repo="utils/llm_utils.py",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="utils/openai_utils.py",
        path_in_repo="utils/openai_utils.py",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="utils/__init__.py",
        path_in_repo="utils/__init__.py",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="constants.py",
        path_in_repo="constants.py",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj="structures.py",
        path_in_repo="structures.py",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )
    api.upload_file(
        path_or_fileobj=repo_requirements.encode(),
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
    )

    webbrowser.open(f"https://huggingface.co/spaces/{repo_id}")
