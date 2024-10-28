import time

import gradio as gr
from utils.llm_utils import process_input_with_retrieval
from zenml import step


def predict(message, history):
    return process_input_with_retrieval(
        input=message,
        n_items_retrieved=5,
        use_reranking=False,
    )


@step
def gradio_rag_deployment() -> None:
    """Launches a Gradio chat interface with the slow echo demo.

    Starts a web server with a chat interface that echoes back user messages.
    The server runs indefinitely until manually stopped.
    """
    demo = gr.ChatInterface(predict, type="messages")
    demo.launch(share=True, inbrowser=True)
    # Keep the step running
    while True:
        time.sleep(1)
