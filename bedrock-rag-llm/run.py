import json

from botocore.exceptions import ClientError
from rich import print
from zenml import pipeline, step
from zenml.client import Client
from zenml.logger import get_logger
from litellm import completion

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


@step
def basic_inference(model_id: str, prompt: str) -> str:
    # Create an Amazon Bedrock Runtime client.
    zc = Client()
    sc_client = zc.get_service_connector_client(
        name_id_or_prefix="0b04bcae-efc9-4044-a1c2-b86281cb0820",
        resource_type="aws-generic",
    ).connect()

    brt = sc_client.client("bedrock-runtime", region_name="us-east-1")

    try:
        system_prompt = "You are a helpful assistant."
        max_tokens = 1000

        # Prompt with user turn only.
        user_message = {"role": "user", "content": prompt}
        messages = [user_message]

        response = generate_message(
            brt, model_id, system_prompt, messages, max_tokens
        )
        logger.debug(response)

        return response["content"][0]["text"]

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {format(message)}")


@pipeline
def bedrock_basic_inference(model_id: str, prompt: str):
    basic_inference(model_id, prompt)


if __name__ == "__main__":
    bedrock_basic_inference(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        prompt="What is the capital of France?",
    )
