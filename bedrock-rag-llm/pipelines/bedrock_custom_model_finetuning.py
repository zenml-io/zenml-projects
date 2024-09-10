import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated

import jsonlines
import tiktoken
from constants import (
    AWS_CUSTOM_MODEL_BUCKET_NAME,
    AWS_CUSTOM_MODEL_CUSTOMIZATION_TYPE,
    AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME,
    AWS_CUSTOM_MODEL_ROLE_ARN,
    AWS_REGION,
    AWS_SERVICE_CONNECTOR_ID,
    MODEL_DEFINITION,
)
from zenml import log_model_metadata, pipeline, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.service_connectors.service_connector import ServiceConnector

logger = get_logger(__name__)

warnings.filterwarnings("ignore")


def get_boto_client() -> ServiceConnector:
    logger.info("Getting boto client...")
    zc = Client()
    return zc.get_service_connector_client(
        name_id_or_prefix=AWS_SERVICE_CONNECTOR_ID,
        resource_type="aws-generic",
    ).connect()


def upload_to_s3(file_path: str, bucket_name: str, object_key: str) -> None:
    logger.info(
        f"Uploading file to S3: {file_path} -> s3://{bucket_name}/{object_key}"
    )
    boto_client = get_boto_client()
    s3_client = boto_client.client("s3")
    s3_client.upload_file(file_path, bucket_name, object_key)
    logger.info("File uploaded successfully")


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
def load_split_push_data_to_s3(dataset_dir: str) -> Annotated[str, "raw_data"]:
    logger.info(f"Processing dataset from directory: {dataset_dir}")
    data = []
    for path in Path(dataset_dir).rglob("*.md"):
        with open(path, "r") as file:
            text = file.read()
            chunks = split_text(text)
            data.extend({"input": chunk} for chunk in chunks)

    logger.info(f"Total chunks processed: {len(data)}")

    tmp_file_path = Path("/tmp") / AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME
    logger.info(f"Writing data to temporary file: {tmp_file_path}")
    with jsonlines.open(tmp_file_path, mode="w") as writer:
        writer.write_all(data)

    upload_to_s3(
        tmp_file_path,
        AWS_CUSTOM_MODEL_BUCKET_NAME,
        AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME,
    )

    total_tokens = sum(count_tokens(chunk["input"]) for chunk in data)
    logger.info(f"Total tokens in dataset: {total_tokens}")

    log_model_metadata(
        metadata={
            "finetuning": {
                "finetuning_num_documents": len(data),
                "finetuning_num_tokens": total_tokens,
                "finetuning_data_s3_uri": f"s3://{AWS_CUSTOM_MODEL_BUCKET_NAME}/{AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME}",
                "finetuning_data_s3_bucket": AWS_CUSTOM_MODEL_BUCKET_NAME,
                "finetuning_data_s3_object_key": AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME,
            }
        }
    )
    logger.info("Model metadata logged")

    return "".join(chunk["input"] for chunk in data)


@step
def submit_model_customization_request(
    base_model_identifier: str = "amazon.titan-text-express-v1:0:8k",
) -> str:
    logger.info(
        f"Submitting model customization request for base model: {base_model_identifier}"
    )
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    boto_client = get_boto_client()
    bedrock_client = boto_client.client("bedrock", region_name=AWS_REGION)

    learning_rate = 0.00001
    epoch_count = 5
    batch_size = 1
    learning_rate_warmup_steps = 5

    logger.info("Creating model customization job...")
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
            "learningRate": str(learning_rate),
            "epochCount": str(epoch_count),
            "batchSize": str(batch_size),
            "learningRateWarmupSteps": str(learning_rate_warmup_steps),
        },
    )

    job_arn = response["jobArn"]
    logger.info(f"Finetuning job ARN: {job_arn}")
    log_model_metadata(
        metadata={
            "finetuning": {
                "customization_type": AWS_CUSTOM_MODEL_CUSTOMIZATION_TYPE,
                "finetuning_base_model_identifier": base_model_identifier,
                "learning_rate": learning_rate,
                "epoch_count": epoch_count,
                "batch_size": batch_size,
                "learning_rate_warmup_steps": learning_rate_warmup_steps,
            }
        }
    )
    logger.info("Model customization metadata logged")
    return job_arn


@pipeline(model=MODEL_DEFINITION)
def bedrock_custom_model_finetuning(dataset_dir: str):
    _ = load_split_push_data_to_s3(dataset_dir)
    submit_model_customization_request(after="load_split_push_data_to_s3")
