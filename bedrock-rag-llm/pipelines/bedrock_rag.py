from steps.create_and_sync_knowledge_base import create_and_sync_knowledge_base
from steps.evaluate_rag import evaluate_rag
from steps.load_and_push_data_to_s3 import load_and_push_data_to_s3
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def bedrock_rag():
    load_and_push_data_to_s3(bucket_name="bedrock-zenml-rag-docs")
    kb_id = create_and_sync_knowledge_base(after="load_and_push_data_to_s3")
    evaluate_rag(kb_id)
