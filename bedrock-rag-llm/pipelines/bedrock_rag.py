from steps.load_and_push_data_to_s3 import load_and_push_data_to_s3
from zenml import pipeline, step


@step
def create_and_sync_knowledge_base():
    pass


@step
def evaluate_rag():
    pass


@pipeline
def bedrock_rag():
    load_and_push_data_to_s3(bucket_name="bedrock-zenml-rag-docs")
    create_and_sync_knowledge_base()
    evaluate_rag()
