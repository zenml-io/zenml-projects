import json

from zenml.logger import get_logger

logger = get_logger(__name__)


def generate_message(
    bedrock_runtime, model_id, system_prompt, messages, max_tokens
):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
    )

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    return json.loads(response.get("body").read())
