from constants import AWS_REGION, CLAUDE_3_HAIKU_MODEL_ARN
from zenml import step


@step
def evaluate_rag(knowledge_base_id: str) -> str:
    bedrock_agent_runtime_client = boto3_session.client(
        "bedrock-agent-runtime", region_name=AWS_REGION
    )
    query = "What orchestrators does ZenML support?"
    logger.info(f"Evaluating RAG with query: {query}")

    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": knowledge_base_id,
                "modelArn": CLAUDE_3_HAIKU_MODEL_ARN,
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
        knowledgeBaseId=knowledge_base_id,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 3  # will fetch top 3 documents which matches closely with the query.
            }
        },
    )

    logger.info("Printing out relevant documents:")
    for doc in relevant_documents["retrievalResults"]:
        logger.info(doc["content"]["text"])
