import logging
import os
from typing import List

import faiss
from langchain.docstore.document import Document
from langchain.document_loaders import GitbookLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from llama_index import Document as LlamaDocument
from llama_index import GPTFaissIndex, download_loader
from slack_sdk import WebClient
from zenml.integrations.registry import integration_registry
from zenml.pipelines import pipeline
from zenml.post_execution import get_pipeline
from zenml.steps import BaseParameters, step


def get_channel_id_from_name(name: str) -> str:
    """Gets a channel ID from a Slack channel name.

    Args:
        name: Name of the channel.

    Returns:
        Channel ID.
    """
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    response = client.conversations_list()
    conversations = response["channels"]
    if id := [c["id"] for c in conversations if c["name"] == name][0]:
        return id
    else:
        raise ValueError(f"Channel {name} not found.")


SLACK_CHANNEL_IDS = [get_channel_id_from_name("general")]


@pipeline
def docs_to_index_pipeline(document_loader, slack_loader, index_generator):
    slack_docs = slack_loader()
    documents = document_loader()
    index_generator(documents, slack_docs)


class DocsLoaderParameters(BaseParameters):
    docs_uri: str = "https://docs.zenml.io"


@step
def docs_loader(params: DocsLoaderParameters) -> List[Document]:
    # loader = GitbookLoader(params.docs_uri)
    # page_data = loader.load()
    loader = GitbookLoader(params.docs_uri, load_all_paths=True)
    all_pages_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(all_pages_data)


class SlackLoaderParameters(BaseParameters):
    channel_ids: List[str] = SLACK_CHANNEL_IDS


@step
def slack_loader(params: SlackLoaderParameters) -> List[Document]:
    SlackReader = download_loader("SlackReader")
    loader = SlackReader(slack_token=os.environ["SLACK_BOT_TOKEN"])
    documents = loader.load_data(channel_ids=params.channel_ids)
    [d.to_langchain_format() for d in documents]
    return documents


@step
def index_generator(documents: List[Document]) -> VectorStore:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


@step
def llama_index_generator(
    documents: List[Document], slack_documents: List[Document]
) -> GPTFaissIndex:
    documents = [LlamaDocument.from_langchain_format(d) for d in documents]
    documents.extend(slack_documents)
    faiss_index = faiss.IndexFlatL2(1536)
    return GPTFaissIndex(documents, faiss_index=faiss_index)


def run_langchain():
    pipeline = docs_to_index_pipeline(
        document_loader=docs_loader(),
        index_generator=index_generator(),
    )
    pipeline.configure(enable_cache=False)
    pipeline.run()


def run_llama():
    pipeline = docs_to_index_pipeline(
        document_loader=docs_loader(),
        slack_loader=slack_loader(),
        index_generator=llama_index_generator(),
    )
    pipeline.configure(enable_cache=False)
    pipeline.run()


def post_exec_llama_index():
    integration_registry.activate_integrations()
    pipeline = get_pipeline("docs_to_index_pipeline")
    last_run = pipeline.runs[0]
    index = last_run.get_step("index_generator").output.read()
    response = index.query(
        "Are materializers loaded automatically during the post execution workflow?"
    )
    print(response)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    run_llama()
    post_exec_llama_index()
