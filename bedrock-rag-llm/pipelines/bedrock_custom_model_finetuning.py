import json
from pathlib import Path

import tiktoken
from constants import AWS_SERVICE_CONNECTOR_ID
from zenml import pipeline, step
from zenml.client import Client
from zenml.service_connectors.service_connector import ServiceConnector

pretraining_data_filename = "pretraining_inputs.jsonl"
bucket_name = "bedrock-zenml-rag-docs"


def get_boto_client() -> ServiceConnector:
    zc = Client()
    return zc.get_service_connector_client(
        name_id_or_prefix=AWS_SERVICE_CONNECTOR_ID,
        resource_type="aws-generic",
    ).connect()


def upload_to_s3(file_path: str, bucket_name: str, object_key: str) -> None:
    boto_client = get_boto_client()
    s3_client = boto_client.client("s3")
    s3_client.upload_file(file_path, bucket_name, object_key)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_text(text, max_tokens=600, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start = end
    return chunks


@step
def load_split_push_data_to_s3(dataset_dir: str) -> str:
    data = []
    for path in Path(dataset_dir).rglob("*.md"):
        with open(path, "r") as file:
            text = file.read()
            chunks = split_text(text)
            data.extend(
                json.dumps({"input": chunk}) + "\n" for chunk in chunks
            )
    tmp_file_path = Path("/tmp") / pretraining_data_filename
    with open(tmp_file_path, "w") as f:
        f.writelines(data)

    upload_to_s3(tmp_file_path, bucket_name, pretraining_data_filename)

    return "".join(data)


@step
def finetune_model():
    pass


@pipeline
def bedrock_custom_model_finetuning(dataset_dir: str):
    _ = load_split_push_data_to_s3(dataset_dir)
    finetune_model(after="load_split_push_data_to_s3")
