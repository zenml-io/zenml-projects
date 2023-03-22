# Adapted from the `MrsStax` repo https://github.com/normandmickey/MrsStax

# To run this slackblot, execute `python run.py`. If you want to use a model
# from the HuggingFaceHub instead of OpenAI, you can pass in an argument via the
# command line as in `python run.py --model huggingface` and customise the code
# as appropriate.

import argparse
import os

from langchain import HuggingFaceHub, OpenAI, PromptTemplate
from langchain.chains import ChatVectorDBChain, SequentialChain
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from zenml.enums import ExecutionStatus
from zenml.post_execution import get_pipeline

# TODO: These should be saved in the Secret Store
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="openai", choices=["openai", "huggingface"]
)
args = parser.parse_args()

openai_llm = OpenAI(temperature=0, max_tokens=500)
huggingface_llm = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs={"temperature": 0, "max_length": 64},
)

if args.model == "openai":
    llm = openai_llm
elif args.model == "huggingface":
    llm = huggingface_llm
else:
    raise ValueError(f"Invalid model argument: {args.model}")


def get_vector_store(version: str):
    pipeline = get_pipeline("zenml_docs_index_generation")
    runs = pipeline.runs
    for run_ in runs:
        if run_.status != ExecutionStatus.COMPLETED:
            continue
        if run_.name.split("_")[-1] == version:
            return run_.steps[-1].output.read()
    raise RuntimeError(
        "No index versions found. Please run `python run.py` first."
    )


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

vector_store = get_vector_store("0.35.1")

chatgpt_chain = ChatVectorDBChain.from_llm(llm=llm, vectorstore=vector_store)


seq_chain = SequentialChain(
    chains=[chatgpt_chain], input_variables=["chat_history", "question"]
)


@app.message(".*")
def message_handler(message, say, logger):
    print(message)

    output = seq_chain.run(
        chat_history="", question=message["text"], verbose=True
    )
    say(output)


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
