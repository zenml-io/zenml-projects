from constants import MODEL_DEFINITION
from steps.create_and_sync_knowledge_base import (
    create_and_sync_knowledge_base,
    create_aoss_collection,
    create_vector_index,
)
from steps.evaluate_rag import evaluate_rag, visualize_rag_scores
from steps.load_and_push_data_to_s3 import load_and_push_data_to_s3
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline(model=MODEL_DEFINITION)
def bedrock_rag():
    load_and_push_data_to_s3(bucket_name="bedrock-zenml-rag-docs")
    host, collection_id, collection_arn = create_aoss_collection()
    create_vector_index(host, collection_id)
    kb_id = create_and_sync_knowledge_base(
        collection_arn,
        collection_id,
        after=["load_and_push_data_to_s3", "create_vector_index"],
    )
    bedrock_scores, base_model_scores = evaluate_rag(kb_id)
    visualize_rag_scores(bedrock_scores, base_model_scores)
