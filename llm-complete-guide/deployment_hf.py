import logging
import os

import gradio as gr
from constants import SECRET_NAME
from utils.llm_utils import process_input_with_retrieval
from zenml.client import Client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_ENVIRONMENT = os.getenv("GRADIO_ZENML_APP_ENVIRONMENT", "dev")

# Initialize ZenML client and verify secret access
try:
    client = Client()
    secret = client.get_secret(SECRET_NAME)
    logger.info(
        f"Successfully initialized ZenML client and found secret {SECRET_NAME}"
    )
except Exception as e:
    logger.error(f"Failed to initialize ZenML client or access secret: {e}")
    raise RuntimeError(f"Application startup failed: {e}")


# def predict(message, history):
#     try:
#         return process_input_with_retrieval(
#             input=message,
#             n_items_retrieved=20,
#             use_reranking=True,
#             tracing_tags=["gradio", "web-interface", APP_ENVIRONMENT],
#         )
#     except Exception as e:
#         logger.error(f"Error processing message: {e}")
#         return f"Sorry, I encountered an error: {str(e)}"


# # Launch the Gradio interface
# interface = gr.ChatInterface(
#     predict,
#     title="ZenML Documentation Assistant",
#     description="Ask me anything about ZenML!",
# )


def vote(data: gr.LikeData):
    """Vote on a response.

    Args:
        data (gr.LikeData): The vote data.
    """
    trace = logger.info("Getting trace from Langfuse")

    logger.info(f"Vote data: {data}")
    if data.liked:
        logger.info("Vote up")
    else:
        logger.info("Vote down")


def predict(message, history):
    try:
        # add the prod flag here
        return process_input_with_retrieval(
            input=message,
            n_items_retrieved=20,
            use_reranking=True,
            model_version_stage="production",
            tracing_tags=["gradio", "web-interface", APP_ENVIRONMENT],
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


with gr.Blocks() as interface:
    custom_chatbot = gr.Chatbot(
        type="messages",
        editable=True,
    )

    gr.ChatInterface(
        predict,
        type="messages",
        title="ZenML Documentation Assistant",
        description="Ask me anything about ZenML!",
        chatbot=custom_chatbot,
        theme="shivi/calm_seafoam",
    )

    custom_chatbot.like(vote, None, None)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", share=False)
