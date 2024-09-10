import json
import time
import warnings

from constants import (
    AWS_BEDROCK_KB_EXECUTION_ROLE_ARN,
    AWS_REGION,
    AWS_SERVICE_CONNECTOR_ID,
)
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from retrying import retry
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.service_connectors.service_connector import ServiceConnector

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


def get_boto_client() -> ServiceConnector:
    zc = Client()
    return zc.get_service_connector_client(
        name_id_or_prefix=AWS_SERVICE_CONNECTOR_ID,
        resource_type="aws-generic",
    ).connect()


# suffix = random.randrange(200, 900)
suffix = 504

logger.info("Initializing AWS clients...")
boto3_session = get_boto_client()
iam_client = boto3_session.client("iam", region_name=AWS_REGION)
account_number = (
    boto3_session.client("sts", region_name=AWS_REGION)
    .get_caller_identity()
    .get("Account")
)
identity = boto3_session.client(
    "sts", region_name=AWS_REGION
).get_caller_identity()["Arn"]
sts_client = boto3_session.client("sts", region_name=AWS_REGION)

bedrock_agent_client = boto3_session.client(
    "bedrock-agent", region_name=AWS_REGION
)
bedrock_agent_runtime_client = boto3_session.client(
    "bedrock-agent-runtime", region_name=AWS_REGION
)

service = "aoss"
s3_client = boto3_session.client("s3")
account_id = sts_client.get_caller_identity()["Account"]
s3_suffix = f"{AWS_REGION}-{account_id}"

logger.info("Setting up policy and role names...")
encryption_policy_name = f"bedrock-sample-rag-sp-{suffix}"
network_policy_name = f"bedrock-sample-rag-np-{suffix}"
access_policy_name = f"bedrock-sample-rag-ap-{suffix}"
bedrock_execution_role_name = (
    f"AmazonBedrockExecutionRoleForKnowledgeBase_{suffix}"
)
fm_policy_name = f"AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}"
s3_policy_name = f"AmazonBedrockS3PolicyForKnowledgeBase_{suffix}"
sm_policy_name = f"AmazonBedrockSecretPolicyForKnowledgeBase_{suffix}"
oss_policy_name = f"AmazonBedrockOSSPolicyForKnowledgeBase_{suffix}"

bucket_name = "bedrock-zenml-rag-docs"

vector_store_name = f"bedrock-vectordb-rag-{suffix}"
index_name = f"bedrock-vectordb-rag-index-{suffix}"
aoss_client = boto3_session.client(
    "opensearchserverless", region_name=AWS_REGION
)

bedrock_kb_execution_role_arn = AWS_BEDROCK_KB_EXECUTION_ROLE_ARN


@retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
def create_knowledge_base_func():
    logger.info("Creating knowledge base...")
    create_kb_response = bedrock_agent_client.create_knowledge_base(
        name=name,
        description=description,
        roleArn=roleArn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embeddingModelArn
            },
        },
        storageConfiguration={
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": opensearchServerlessConfiguration,
        },
    )
    return create_kb_response["knowledgeBase"]


@step
def create_and_sync_knowledge_base(
    kb_name: str = vector_store_name,
    kb_description: str = "Bedrock RAG",
    role_arn: str = bedrock_kb_execution_role_arn,
) -> str:
    logger.info("Starting creation of knowledge base and synchronization...")
    try:
        logger.info("Creating collection using AWS OpenSearch Serverless...")
        collection = aoss_client.create_collection(
            name=vector_store_name, type="VECTORSEARCH"
        )
        collection_id = collection["createCollectionDetail"]["id"]
    except Exception as err:
        if err.response["Error"]["Code"] == "ConflictException":
            logger.info("Collection already exists. Skipping creation.")
            collections = aoss_client.list_collections(maxResults=15)
            collection = next(
                (c for c in collections["collectionSummaries"] if c["name"] == vector_store_name),
                None
            )
            collection_id = collection["id"]
            if collection is None:
                raise ValueError(f"Collection {vector_store_name} not found") from err

    host = f"{collection_id}.{AWS_REGION}.aoss.amazonaws.com"

    logger.info("Creating vector index...")
    credentials = get_boto_client().get_credentials()
    awsauth = auth = AWSV4SignerAuth(credentials, AWS_REGION, service)

    breakpoint()
    index_name = f"bedrock-sample-index-{suffix}"
    body_json = {
        "settings": {"index.knn": "true"},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "l2",
                        "parameters": {"ef_construction": 200, "m": 16},
                    },
                },
                "text": {"type": "text"},
                "text-metadata": {"type": "text"},
            }
        },
    }

    logger.info("Building OpenSearch client...")
    oss_client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300,
    )
    logger.info("Waiting for data access rules to be enforced...")
    time.sleep(60)

    logger.info("Creating index...")
    response = oss_client.indices.create(
        index=index_name, body=json.dumps(body_json)
    )

    logger.info("Setting up knowledge base configuration...")
    opensearchServerlessConfiguration = {
        "collectionArn": collection["createCollectionDetail"]["arn"],
        "vectorIndexName": index_name,
        "fieldMapping": {
            "vectorField": "vector",
            "textField": "text",
            "metadataField": "text-metadata",
        },
    }

    chunkingStrategyConfiguration = {
        "chunkingStrategy": "FIXED_SIZE",
        "fixedSizeChunkingConfiguration": {
            "maxTokens": 512,
            "overlapPercentage": 20,
        },
    }

    s3Configuration = {
        "bucketArn": f"arn:aws:s3:::{bucket_name}",
    }

    embeddingModelArn = f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v1"

    name = f"bedrock-sample-knowledge-base-{suffix}"
    description = "Bedrock Knowledge Bases for Web URL and S3 Connector"
    roleArn = bedrock_kb_execution_role_arn

    logger.info("Creating knowledge base...")
    try:
        kb = create_knowledge_base_func()
    except Exception as err:
        logger.error(f"Error creating knowledge base: {err=}, {type(err)=}")

    kb_id = kb["knowledgeBaseId"]

    logger.info("Retrieving knowledge base...")
    get_kb_response = bedrock_agent_client.get_knowledge_base(
        knowledgeBaseId=kb_id
    )

    logger.info("Creating S3 DataSource in KnowledgeBase...")
    create_ds_response = bedrock_agent_client.create_data_source(
        name=name,
        description=description,
        knowledgeBaseId=kb["knowledgeBaseId"],
        dataDeletionPolicy="DELETE",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": s3Configuration,
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": chunkingStrategyConfiguration
        },
    )

    ds = create_ds_response["dataSource"]

    logger.info("Retrieving S3 datasource...")
    bedrock_agent_client.get_data_source(
        knowledgeBaseId=kb["knowledgeBaseId"], dataSourceId=ds["dataSourceId"]
    )

    time.sleep(10)

    logger.info("Starting ingestion job...")
    start_job_response = bedrock_agent_client.start_ingestion_job(
        knowledgeBaseId=kb["knowledgeBaseId"], dataSourceId=ds["dataSourceId"]
    )

    job = start_job_response["ingestionJob"]

    logger.info("Monitoring ingestion job status...")
    while job["status"] != "COMPLETE":
        get_job_response = bedrock_agent_client.get_ingestion_job(
            knowledgeBaseId=kb["knowledgeBaseId"],
            dataSourceId=ds["dataSourceId"],
            ingestionJobId=job["ingestionJobId"],
        )
        job = get_job_response["ingestionJob"]
        logger.info(f"Ingestion job status: {job['status']}")
        time.sleep(10)

    logger.info("Ingestion job completed successfully")

    return kb["knowledgeBaseId"]
