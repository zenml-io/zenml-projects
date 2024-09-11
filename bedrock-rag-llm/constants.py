from zenml import Model
from zenml.model.model import ModelStages

CLAUDE_3_HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
CLAUDE_3_HAIKU_MODEL_ARN = (
    f"arn:aws:bedrock:us-east-1::foundation-model/{CLAUDE_3_HAIKU_MODEL_ID}"
)

AWS_SERVICE_CONNECTOR_ID = "0b04bcae-efc9-4044-a1c2-b86281cb0820"

AWS_REGION = "us-east-1"
AWS_CUSTOM_MODEL_ROLE_ARN = (
    "arn:aws:iam::339712793861:role/AmazonBedrockCustomizationRole1"
)
AWS_CUSTOM_MODEL_CUSTOMIZATION_TYPE = "CONTINUED_PRE_TRAINING"
AWS_CUSTOM_MODEL_BUCKET_NAME = "bedrock-zenml-rag-docs"
AWS_CUSTOM_MODEL_PRETRAINING_DATA_FILENAME = "pretraining_inputs.jsonl"
AWS_BEDROCK_KB_EXECUTION_ROLE_ARN = (
    "AmazonBedrockExecutionRoleForKnowledgeBase_392"
    # "AmazonBedrockExecutionRoleForKnowledgeBase_96gjm"
)


MODEL_DEFINITION = Model(
    name="aws-bedrock-doordash-usa",
    description="DoorDash's integration of AWS Bedrock and Amazon Connect for enhancing self-service offerings in their contact center",
    audience="DoorDash customer support staff and developers working on improving customer service efficiency",
    use_cases="Use this model to power generative AI-driven self-service solutions, enhancing customer support interactions and efficiency",
    limitations="The model's effectiveness may vary depending on the complexity of customer queries and specific food delivery industry jargon",
    trade_offs="Improved customer support efficiency through AI-driven self-service may require ongoing model updates to accommodate new types of customer inquiries",
    tags=[
        "bedrock",
        "doordash",
        "generative-ai",
        "customer-support",
        "food-delivery",
        "aws",
        "amazon-connect",
    ],
    version=ModelStages.LATEST,
)
