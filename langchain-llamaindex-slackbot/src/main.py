# Adapted from the `MrsStax` repo https://github.com/normandmickey/MrsStax

# To run this slackblot, execute `python run.py`. If you want to use a model
# from the HuggingFaceHub instead of OpenAI, you can pass in an argument via the
# command line as in `python run.py --model huggingface` and customize the code
# as appropriate.

import os
from threading import Thread

import uvicorn
from fastapi import FastAPI
from langchain import HuggingFaceHub, OpenAI, PromptTemplate
from langchain.chains import ChatVectorDBChain, SequentialChain
from openai.error import InvalidRequestError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slackbot_utils import (
    connect_to_zenml_server,
    convert_to_chat_history,
    get_last_n_messages,
    get_vector_store,
)
from zenml.logger import get_logger

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


logger = get_logger(__name__)

model = "openai"

if model == "openai":
    openai_llm = OpenAI(
        temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo"
    )
    llm = openai_llm
elif model == "huggingface":
    huggingface_llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 0, "max_length": 500},
    )
    llm = huggingface_llm
else:
    raise ValueError(f"Invalid model argument: {model}")


connect_to_zenml_server()


# Initializes your app with your bot token and socket mode handler
app = App(token=SLACK_BOT_TOKEN)

# Langchain implementation
template = """
    Using only the following context answer the question at the end. If you can't find the answer in the context below, just say that you don't know. Do not make up an answer.
    {chat_history}
    Human: {question}
    Assistant:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "question"], template=template
)

vector_store = get_vector_store()

chatgpt_chain = ChatVectorDBChain.from_llm(llm=llm, vectorstore=vector_store)


@app.event({"type": "message", "subtype": None})
def reply_in_thread(body: dict, say, context):
    """Listens to messages and replies in a thread.

    Args:
        body (dict): Slack event body
        say (function): Slack say function
        context (dict): Slack context
    """
    event = body["event"]
    thread_ts = event.get("thread_ts", None) or event["ts"]

    if context["bot_user_id"] in event["text"]:
        logger.debug(f"Received message: {event['text']}")
        if event.get("thread_ts", None):
            full_thread = [
                f"{msg['text']}"
                for msg in context.client.conversations_replies(
                    channel=context["channel_id"], ts=event["thread_ts"]
                ).data["messages"]
            ]
        else:
            full_thread = []

        seq_chain = SequentialChain(
            chains=[chatgpt_chain],
            input_variables=["chat_history", "question"],
        )
        try:
            output = seq_chain.run(
                chat_history=convert_to_chat_history(full_thread),
                question=event["text"],
                verbose=True,
            )
        except InvalidRequestError as e:
            logger.warning(e)
            output = seq_chain.run(
                chat_history=get_last_n_messages(
                    convert_to_chat_history(full_thread)
                ),
                question=event["text"],
                verbose=True,
            )
        say(text=output, thread_ts=thread_ts)


fast_api_app = FastAPI()


@fast_api_app.get("/health")
def health_check():
    return {"status": "OK"}


def run_fastapi():
    uvicorn.run(fast_api_app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    fastapi_thread = Thread(target=run_fastapi)
    fastapi_thread.start()
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
