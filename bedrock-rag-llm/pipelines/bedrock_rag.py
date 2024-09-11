from constants import MODEL_DEFINITION
from steps.create_and_sync_knowledge_base import create_and_sync_knowledge_base
from steps.evaluate_rag import evaluate_rag, visualize_rag_scores
from steps.load_and_push_data_to_s3 import load_and_push_data_to_s3
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline(model=MODEL_DEFINITION)
def bedrock_rag():
    load_and_push_data_to_s3(bucket_name="bedrock-zenml-rag-docs")
    kb_id = create_and_sync_knowledge_base(after="load_and_push_data_to_s3")
    bedrock_scores, base_model_scores = evaluate_rag(kb_id)
    visualize_rag_scores(bedrock_scores, base_model_scores)
