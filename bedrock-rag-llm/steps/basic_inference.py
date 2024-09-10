from botocore.exceptions import ClientError
from constants import AWS_SERVICE_CONNECTOR_ID
from rich import print
from utils import generate_message
from zenml import log_model_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def basic_inference(model_id: str, prompt: str) -> str:
    # Create an Amazon Bedrock Runtime client.
    zc = Client()
    sc_client = zc.get_service_connector_client(
        name_id_or_prefix=AWS_SERVICE_CONNECTOR_ID,
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

        log_model_metadata(
            metadata={
                "basic_inference": {
                    "model_id": model_id,
                    "prompt": prompt,
                    "response": response,
                }
            }
        )

        return response["content"][0]["text"]

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {format(message)}")
