import json

import boto3
from botocore.exceptions import ClientError
from rich import print
from zenml import pipeline, step


@step
def bedrock_basic_inference(model_id: str, prompt: str) -> str:
    # Create an Amazon Bedrock Runtime client.
    brt = boto3.client("bedrock-runtime")

    # Format the request payload using the model's native structure.
    native_request = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.5,
            "topP": 0.9,
        },
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = brt.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["results"][0]["outputText"]
    print(response_text)

    return response_text


@pipeline
def bedrock_basic_inference(model_id: str, prompt: str):
    bedrock_basic_inference(model_id, prompt)


if __name__ == "__main__":
    print(
        bedrock_basic_inference(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            prompt="What is the capital of France?",
        )
    )
