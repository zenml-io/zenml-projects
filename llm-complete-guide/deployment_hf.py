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


def predict(message, history):
    try:
        return process_input_with_retrieval(
            input=message,
            n_items_retrieved=20,
            use_reranking=True,
            tracing_tags=["gradio", "web-interface", APP_ENVIRONMENT],
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"Sorry, I encountered an error: {str(e)}"


# Launch the Gradio interface
interface = gr.ChatInterface(
    predict,
    title="ZenML Documentation Assistant",
    description="Ask me anything about ZenML!",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", share=False)
