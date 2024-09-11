import io
from typing import Annotated, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import PIL
from botocore.exceptions import ClientError
from constants import (
    AWS_REGION,
    AWS_SERVICE_CONNECTOR_ID,
    CLAUDE_3_HAIKU_MODEL_ARN,
    CLAUDE_3_HAIKU_MODEL_ID,
)
from PIL import Image
from utils import generate_message
from zenml import log_model_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

from steps.create_and_sync_knowledge_base import get_boto_client

logger = get_logger(__name__)

bad_answers = [
    {
        "question": "What orchestrators does ZenML support?",
        "bad_words": ["AWS Step Functions", "Flyte", "Prefect", "Dagster"],
    },
    {
        "question": "What is the default orchestrator in ZenML?",
        "bad_words": ["Flyte", "AWS Step Functions"],
    },
    {
        "question": "What model deployers do you support?",
        "bad_words": ["Baseten", "VertexAI"],
    },
]

bad_immediate_responses = [
    {
        "question": "Does ZenML support the Flyte orchestrator out of the box?",
        "bad_immediate_response": ["Yes"],
    },
    {
        "question": "What is the default orchestrator in ZenML?",
        "bad_immediate_response": ["The Kubernetes orchestrator"],
    },
    {
        "question": "Can I write my own custom orchestrator or other stack components in ZenML?",
        "bad_immediate_response": ["No"],
    },
    {
        "question": "Is ZenML suitable for production use?",
        "bad_immediate_response": ["No"],
    },
    {
        "question": "Is ZenML suitable for GenAI use cases?",
        "bad_immediate_response": ["No"],
    },
]

good_responses = [
    {
        "question": "What are the supported orchestrators in ZenML? Please list as many of the supported ones as possible.",
        "good_words": ["Kubeflow", "Airflow"],
    },
    {
        "question": "What is the default orchestrator in ZenML?",
        "good_words": ["local"],
    },
    {
        "question": "Can I write my own custom orchestrator or other stack components in ZenML?",
        "good_words": ["Yes"],
    },
]


def evaluate_bad_answers(
    bad_answers: List[Dict[str, Union[str, List[str]]]],
    bedrock_agent_runtime_client,
    knowledge_base_id: str,
    base_model: bool = False,
) -> Annotated[float, "bad_answers_score"]:
    logger.info("Starting evaluation of bad answers")
    total_bad_answers = len(bad_answers)
    bad_answers_count = 0

    for example in bad_answers:
        question = example["question"]
        bad_words = example["bad_words"]

        logger.debug(f"Evaluating question: {question}")
        # Perform inference to get the model's response
        if base_model:
            generated_text = base_model_inference(question)
        else:
            model_response = bedrock_inference(
                bedrock_agent_runtime_client, knowledge_base_id, question
            )
            generated_text = model_response["output"]["text"]

        # Search the generated text for the bad words
        for bad_word in bad_words:
            if bad_word.lower() in generated_text.lower():
                bad_answers_count += 1
                logger.debug(f"Bad word '{bad_word}' found in response")
                break  # Move to the next question if any bad word is found

    bad_answers_score = bad_answers_count / total_bad_answers
    percentage_value = bad_answers_score * 100
    logger.info(
        f"Bad answers evaluation completed. Score: {percentage_value}%"
    )
    logger.debug(
        f"Total bad answers: {total_bad_answers}, Bad answers found: {bad_answers_count}"
    )
    return percentage_value


def evaluate_bad_immediate_responses(
    bad_immediate_responses: List[Dict[str, Union[str, List[str]]]],
    bedrock_agent_runtime_client,
    knowledge_base_id: str,
    base_model: bool = False,
) -> Annotated[float, "bad_immediate_responses_score"]:
    logger.info("Starting evaluation of bad immediate responses")
    total_bad_responses = len(bad_immediate_responses)
    bad_responses_count = 0

    for example in bad_immediate_responses:
        question = example["question"]
        bad_immediate_response = example["bad_immediate_response"]

        logger.debug(f"Evaluating question: {question}")
        # Perform inference to get the model's response
        if base_model:
            generated_text = base_model_inference(question)
        else:
            model_response = bedrock_inference(
                bedrock_agent_runtime_client, knowledge_base_id, question
            )
            generated_text = model_response["output"]["text"]

        # Check if the generated text starts with any of the bad immediate responses
        if any(
            generated_text.lower().startswith(response.lower())
            for response in bad_immediate_response
        ):
            bad_responses_count += 1
            logger.debug("Bad immediate response detected")

    bad_responses_score = bad_responses_count / total_bad_responses
    percentage_value = bad_responses_score * 100
    logger.info(
        f"Bad immediate responses evaluation completed. Score: {percentage_value}%"
    )
    logger.debug(
        f"Total bad immediate responses: {total_bad_responses}, Bad responses found: {bad_responses_count}"
    )
    return percentage_value


def evaluate_good_responses(
    good_responses: List[Dict[str, Union[str, List[str]]]],
    bedrock_agent_runtime_client,
    knowledge_base_id: str,
    base_model: bool = False,
) -> Annotated[float, "good_responses_score"]:
    logger.info("Starting evaluation of good responses")
    total_good_responses = len(good_responses)
    good_responses_count = 0

    for example in good_responses:
        question = example["question"]
        good_words = example["good_words"]

        logger.debug(f"Evaluating question: {question}")
        # Perform inference to get the model's response
        if base_model:
            generated_text = base_model_inference(question)
        else:
            model_response = bedrock_inference(
                bedrock_agent_runtime_client, knowledge_base_id, question
            )
            generated_text = model_response["output"]["text"]

        # Check if the generated text contains any of the good words
        if any(
            good_word.lower() in generated_text.lower()
            for good_word in good_words
        ):
            good_responses_count += 1
            logger.debug("Good response detected")

    good_responses_score = good_responses_count / total_good_responses
    percentage_value = good_responses_score * 100
    logger.info(
        f"Good responses evaluation completed. Score: {percentage_value}%"
    )
    logger.debug(
        f"Total good responses: {total_good_responses}, Good responses found: {good_responses_count}"
    )
    return percentage_value


def bedrock_inference(
    bedrock_agent_runtime_client,
    knowledge_base_id: str,
    query: str,
    model: str = CLAUDE_3_HAIKU_MODEL_ARN,
) -> Dict[str, str]:
    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": knowledge_base_id,
                "modelArn": model,
            },
        },
    )
    return response


def base_model_inference(
    query: str,
    model_id: str = CLAUDE_3_HAIKU_MODEL_ID,
) -> str:
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
        user_message = {"role": "user", "content": query}
        messages = [user_message]

        response = generate_message(
            brt, model_id, system_prompt, messages, max_tokens
        )
        logger.debug(response)

        return response["content"][0]["text"]

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)


@step
def evaluate_rag(
    knowledge_base_id: str,
) -> Tuple[
    Annotated[Dict[str, float], "bedrock_scores"],
    Annotated[Dict[str, float], "base_model_scores"],
]:
    boto3_session = get_boto_client()
    bedrock_agent_runtime_client = boto3_session.client(
        "bedrock-agent-runtime", region_name=AWS_REGION
    )

    query = "What orchestrators does ZenML support?"
    logger.info(f"Evaluating RAG with query: {query}")

    model_response = bedrock_inference(
        bedrock_agent_runtime_client, knowledge_base_id, query
    )

    generated_text = model_response["output"]["text"]
    logger.info(f"Generated text: {generated_text}")

    logger.debug("Logging source attributions:")
    citations = model_response["citations"]
    contexts = []
    for citation in citations:
        retrievedReferences = citation["retrievedReferences"]
        contexts.extend(
            reference["content"]["text"] for reference in retrievedReferences
        )
    logger.debug(contexts)

    logger.debug("Retrieving relevant documents:")
    relevant_documents = bedrock_agent_runtime_client.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=knowledge_base_id,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 3  # will fetch top 3 documents which matches closely with the query.
            }
        },
    )

    logger.debug("Printing out relevant documents:")
    for doc in relevant_documents["retrievalResults"]:
        logger.debug(doc["content"]["text"])

    logger.info("Running evals for Bedrock RAG system...")
    bedrock_bad_answers_score = evaluate_bad_answers(
        bad_answers, bedrock_agent_runtime_client, knowledge_base_id
    )
    bedrock_bad_immediate_responses_score = evaluate_bad_immediate_responses(
        bad_immediate_responses,
        bedrock_agent_runtime_client,
        knowledge_base_id,
    )
    bedrock_good_responses_score = evaluate_good_responses(
        good_responses, bedrock_agent_runtime_client, knowledge_base_id
    )

    logger.info("Running comparison evals for base model...")
    base_model_bad_answers_score = evaluate_bad_answers(
        bad_answers,
        base_model_inference,
        knowledge_base_id,
        base_model=True,
    )
    base_model_bad_immediate_responses_score = (
        evaluate_bad_immediate_responses(
            bad_immediate_responses,
            base_model_inference,
            knowledge_base_id,
            base_model=True,
        )
    )
    base_model_good_responses_score = evaluate_good_responses(
        good_responses,
        base_model_inference,
        knowledge_base_id,
        base_model=True,
    )

    log_model_metadata(
        metadata={
            "bedrock_rag": {
                "knowledge_base_id": knowledge_base_id,
                "query": query,
                "generated_text": generated_text,
                "model_arn": CLAUDE_3_HAIKU_MODEL_ARN,
                "bedrock_bad_answers_score": bedrock_bad_answers_score,
                "bedrock_bad_immediate_responses_score": bedrock_bad_immediate_responses_score,
                "bedrock_good_responses_score": bedrock_good_responses_score,
                "base_model_bad_answers_score": base_model_bad_answers_score,
                "base_model_bad_immediate_responses_score": base_model_bad_immediate_responses_score,
                "base_model_good_responses_score": base_model_good_responses_score,
            }
        }
    )

    return (
        {
            "bedrock_bad_answers_score": bedrock_bad_answers_score,
            "bedrock_bad_immediate_responses_score": bedrock_bad_immediate_responses_score,
            "bedrock_good_responses_score": bedrock_good_responses_score,
        },
        {
            "base_model_bad_answers_score": base_model_bad_answers_score,
            "base_model_bad_immediate_responses_score": base_model_bad_immediate_responses_score,
            "base_model_good_responses_score": base_model_good_responses_score,
        },
    )


@step
def visualize_rag_scores(
    bedrock_scores: Dict[str, float],
    base_model_scores: Dict[str, float],
) -> Annotated[PIL.Image.Image, "rag_comparison_scores"]:
    categories = [
        "bad_answers",
        "bad_immediate_responses",
        "good_responses",
    ]
    bedrock_values = [
        bedrock_scores[f"bedrock_{cat}_score"] for cat in categories
    ]
    base_model_values = [
        base_model_scores[f"base_model_{cat}_score"] for cat in categories
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Set the width of each bar and the positions of the bars
    width = 0.35
    y = range(len(categories))

    # Create the bars
    ax.barh(
        [i - width / 2 for i in y],
        bedrock_values,
        width,
        label="Bedrock RAG",
        color="green",
    )
    ax.barh(
        [i + width / 2 for i in y],
        base_model_values,
        width,
        label="Base Model",
        color="blue",
    )

    # Customize the plot
    ax.set_xlabel("Score (%)")
    ax.set_title("RAG Evaluation Scores")
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.legend()

    # Adjust layout and display
    plt.tight_layout()

    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    return img
