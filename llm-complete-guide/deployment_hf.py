import logging
import os
import time

import gradio as gr
from constants import SECRET_NAME
from langfuse import Langfuse
from utils.llm_utils import process_input_with_retrieval
from zenml.client import Client

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


LANGFUSE_PUBLIC_KEY = os.getenv(
    "LANGFUSE_PUBLIC_KEY", secret.secret_values["langfuse_public_key"]
)
LANGFUSE_SECRET_KEY = os.getenv(
    "LANGFUSE_SECRET_KEY", secret.secret_values["langfuse_secret_key"]
)
LANGFUSE_HOST = os.getenv(
    "LANGFUSE_HOST", secret.secret_values["langfuse_host"]
)

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)


def get_langfuse_trace_id() -> str | None:
    """Get the trace from Langfuse.

    This is a very naive implementation. It simply returns the id of the first trace
    in the last 60 seconds. Will retry up to 3 times if no traces are found or if
    there's an error.

    Returns:
        str | None: The trace ID if found, None otherwise
    """
    logger.info("Getting trace from Langfuse")
    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            # Wait 5 seconds before making the API call
            time.sleep(5)
            traces = langfuse.fetch_traces(
                limit=1, order_by="timestamp.desc"
            ).data
            if not traces:
                retries += 1
                if retries == max_retries:
                    logger.error(
                        f"No traces found after {max_retries} attempts"
                    )
                    return None
                logger.warning(
                    f"No traces found (attempt {retries}/{max_retries})"
                )
                time.sleep(10)
                continue
            return traces[0].id
        except Exception as e:
            retries += 1
            if retries == max_retries:
                logger.error(
                    f"Error fetching traces after {max_retries} attempts: {e}"
                )
                return None
            logger.warning(
                f"Error fetching traces (attempt {retries}/{max_retries}): {e}"
            )
            time.sleep(10)
    return None


def vote(data: gr.LikeData):
    """Vote on a response.

    Args:
        data (gr.LikeData): The vote data.
    """

    trace_id = get_langfuse_trace_id()
    logger.info(f"Vote data: {data}")
    if data.liked:
        logger.info("Vote up")
        langfuse.score(
            trace_id=trace_id,
            name="user-explicit-feedback",
            value="like",
            comment="I like this response",
        )
    else:
        logger.info("Vote down")
        langfuse.score(
            trace_id=trace_id,
            name="user-explicit-feedback",
            value="dislike",
            comment="I don't like the response",
        )


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


with gr.Blocks() as interface:
    custom_chatbot = gr.Chatbot(
        type="messages",
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
