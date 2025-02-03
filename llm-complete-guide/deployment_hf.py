import logging
import os
import gradio as gr
from constants import SECRET_NAME
from traceloop.sdk import Traceloop
from utils.llm_utils import process_input_with_retrieval
from zenml.client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    client = Client()
    secret = client.get_secret(SECRET_NAME)
    logger.info(
        f"Successfully initialized ZenML client and found secret {SECRET_NAME}"
    )
except Exception as e:
    logger.error(f"Failed to initialize ZenML client or access secret: {e}")
    raise RuntimeError(f"Application startup failed: {e}")

Traceloop.init(
    # app_name="rag-llm-complete-guide",
    resource_attributes={
        "env": "dev",
        "version": "0.1.0",
    },
    api_key=os.getenv("BRAINTRUST_API_KEY"),
)


def predict(message, history):
    try:
        return process_input_with_retrieval(
            input=message,
            n_items_retrieved=20,
            use_reranking=True,
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
