from zenml import step
import time
import gradio as gr


def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.05)
        yield "You typed: " + message[: i + 1]



@step
def gradio_rag_deployment():
    demo = gr.ChatInterface(slow_echo, type="messages")
    demo.launch()
