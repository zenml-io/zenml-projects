from zenml import step
from zenml.client import Client


@step
def create_and_sync_knowledge_base(
    kb_name: str, kb_description: str, role_arn: str
) -> str:
    zc = Client()
    sc_client = zc.get_service_connector_client(
        name_id_or_prefix="0b04bcae-efc9-4044-a1c2-b86281cb0820",  # TODO: pull this out into config file
        resource_type="aws-generic",
    ).connect()
    brc = sc_client.client("bedrock-agent")
    kb_response = brc.create_knowledge_base(
        name=kb_name,
        description=kb_description,
        roleArn="arn:aws:iam::339712793861:role/service-role/AmazonBedrockExecutionRoleForKnowledgeBase_96gjm",  # TODO: pull this out into config file
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModel": {"name": "amazon.titan-embed-text-v1"}
            },
        },
    )
    kb_id = kb_response["knowledgeBase"]["knowledgeBaseId"]
    return kb_id
