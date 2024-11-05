import gradio as gr
from utils.llm_utils import process_input_with_retrieval


def predict(message, history):
    return process_input_with_retrieval(
        input=message,
        n_items_retrieved=20,
        use_reranking=True,
    )


gr.ChatInterface(predict, type="messages").launch()
