import json
import warnings
from datetime import datetime
from pathlib import Path

import tiktoken
from constants import (
    AWS_CUSTOM_MODEL_BUCKET_NAME,
    AWS_CUSTOM_MODEL_CUSTOMIZATION_TYPE,
    AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME,
    AWS_CUSTOM_MODEL_ROLE_ARN,
    AWS_REGION,
    AWS_SERVICE_CONNECTOR_ID,
)
from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.service_connectors.service_connector import ServiceConnector

logger = get_logger(__name__)

warnings.filterwarnings("ignore")


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
    tmp_file_path = Path("/tmp") / AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME
    with open(tmp_file_path, "w") as f:
        f.writelines(data)

    upload_to_s3(
        tmp_file_path,
        AWS_CUSTOM_MODEL_BUCKET_NAME,
        AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME,
    )

    return "".join(data)


@step
def finetune_model(
    base_model_identifier: str = "amazon.titan-text-express-v1:0:8k",
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    boto_client = get_boto_client()
    bedrock_client = boto_client.client("bedrock", region_name=AWS_REGION)

    response = bedrock_client.create_model_customization_job(
        jobName=f"my-custom-model-finetune-job-{ts}",
        customModelName=f"amazon-titan-text-express-v1-{ts}",
        customizationType=AWS_CUSTOM_MODEL_CUSTOMIZATION_TYPE,
        roleArn=AWS_CUSTOM_MODEL_ROLE_ARN,
        baseModelIdentifier=base_model_identifier,
        jobTags=[{"key": "z-owner", "value": "alex-strick"}],
        customModelTags=[{"key": "z-owner", "value": "alex-strick"}],
        trainingDataConfig={
            "s3Uri": f"s3://{AWS_CUSTOM_MODEL_BUCKET_NAME}/{AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME}"
        },
        outputDataConfig={"s3Uri": f"s3://{AWS_CUSTOM_MODEL_BUCKET_NAME}"},
        hyperParameters={
            "learningRate": "0.00001",
            "epochCount": "5",
            "batchSize": "1",
            "learningRateWarmupSteps": "5",
        },
    )

    job_arn = response["jobArn"]
    logger.info(f"Finetuning job ARN: {job_arn}")
    return job_arn


@pipeline
def bedrock_custom_model_finetuning(dataset_dir: str):
    _ = load_split_push_data_to_s3(dataset_dir)
    finetune_model(after="load_split_push_data_to_s3")
