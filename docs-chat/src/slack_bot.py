# Adapted from the `MrsStax` repo https://github.com/normandmickey/MrsStax

# To run this slackblot, execute `python run.py`. If you want to use a model
# from the HuggingFaceHub instead of OpenAI, you can pass in an argument via the
# command line as in `python run.py --model huggingface` and customise the code
# as appropriate.

import logging
import argparse
import os

from langchain import HuggingFaceHub, OpenAI, PromptTemplate
from langchain.chains import ChatVectorDBChain, SequentialChain
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from zenml.enums import ExecutionStatus
from zenml.post_execution import get_pipeline


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
PIPELINE_NAME = os.getenv("PIPELINE_NAME", "zenml_docs_index_generation")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="openai", choices=["openai", "huggingface"]
)
args = parser.parse_args()


if args.model == "openai":
    openai_llm = OpenAI(temperature=0, max_tokens=500)
    llm = openai_llm
elif args.model == "huggingface":
    huggingface_llm = HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 0, "max_length": 64},
    )
    llm = huggingface_llm
else:
    raise ValueError(f"Invalid model argument: {args.model}")


def get_vector_store():
    """Returns a vector store from latest pipeline run."""
    pipeline = get_pipeline(PIPELINE_NAME)
    for run_ in pipeline.runs:
        if (run_.status == ExecutionStatus.COMPLETED) \
            and run_.status != ExecutionStatus.FAILED:
            # The last step returns the index
            return run_.steps[-1].output.read()

    raise RuntimeError(
        "No index versions found. Please run the pipeline first."
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

@app.message(".*")
def message_handler(message, say, logger):
    logging.info(message)

    vector_store = get_vector_store()

    chatgpt_chain = ChatVectorDBChain.from_llm(llm=llm, vectorstore=vector_store)

    seq_chain = SequentialChain(
        chains=[chatgpt_chain], input_variables=["chat_history", "question"]
    )

    output = seq_chain.run(
        chat_history="", question=message["text"], verbose=True
    )
    say(output)


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
