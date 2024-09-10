from steps.create_and_sync_knowledge_base import create_and_sync_knowledge_base
from steps.load_and_push_data_to_s3 import load_and_push_data_to_s3
from zenml import pipeline, step


@step
def evaluate_rag(knowledge_base_id: str) -> str:
    query = "What orchestrators does ZenML support?"
    logger.info(f"Evaluating RAG with query: {query}")
    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": model_arn,
            },
        },
    )

    generated_text = response["output"]["text"]
    logger.info(f"Generated text: {generated_text}")

    logger.info("Logging source attributions:")
    citations = response["citations"]
    contexts = []
    for citation in citations:
        retrievedReferences = citation["retrievedReferences"]
        contexts.extend(
            reference["content"]["text"] for reference in retrievedReferences
        )
    logger.info(contexts)

    logger.info("Retrieving relevant documents:")
    relevant_documents = bedrock_agent_runtime_client.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=kb_id,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 3  # will fetch top 3 documents which matches closely with the query.
            }
        },
    )

    logger.info("Printing out relevant documents:")
    for doc in relevant_documents["retrievalResults"]:
        logger.info(doc["content"]["text"])


@pipeline
def bedrock_rag():
    load_and_push_data_to_s3(bucket_name="bedrock-zenml-rag-docs")
    kb_id = create_and_sync_knowledge_base()
    evaluate_rag(kb_id)
