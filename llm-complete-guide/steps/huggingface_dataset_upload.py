import polars as pl
from datasets import Dataset
from huggingface_hub import create_repo
from zenml import step
from zenml.client import Client


@step
def upload_chunks_dataset_to_huggingface(
    documents: pl.DataFrame, dataset_suffix: str
) -> str:
    """Uploads chunked documents to Hugging Face dataset."""
    client = Client()
    hf_token = client.get_secret("huggingface_datasets").secret_values["token"]

    repo_name = f"zenml/rag_qa_embedding_questions_{dataset_suffix}"

    create_repo(
        repo_name,
        token=hf_token,
        exist_ok=True,
        private=True,
        repo_type="dataset",
    )

    # Convert the list of questions to a single string
    documents = documents.with_columns(
        pl.col("generated_questions").apply(lambda x: "\n".join(x))
    )

    dataset = Dataset(documents.to_arrow())
    dataset.push_to_hub(
        repo_id=repo_name,
        private=True,
        token=hf_token,
        create_pr=True,
    )
    return repo_name
