import logging
import os
import shutil
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import git
from langchain.docstore.document import Document
from langchain.document_loaders import GitbookLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores import FAISS, VectorStore
from slack_sdk import WebClient
from tqdm import tqdm
from zenml.pipelines import pipeline
from zenml.steps import BaseParameters, step

from slack_reader import SlackReader

# from langchain.vectorstores import Chroma


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


# @pipeline
# def docs_to_index_pipeline(document_loader, slack_loader, index_generator):
#     slack_docs = slack_loader()
#     documents = document_loader()
#     index_generator(documents, slack_docs)


class DocsLoaderParameters(BaseParameters):
    docs_uri: str = "https://docs.zenml.io"
    docs_base_url: str = "https://docs.zenml.io"


@step(enable_cache=True)
def docs_loader(params: DocsLoaderParameters) -> List[Document]:
    # langchain loader; returns langchain documents
    loader = GitbookLoader(
        web_page=params.docs_uri,
        base_url=params.docs_base_url,
        load_all_paths=True,
    )
    all_pages_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(all_pages_data)


class SlackLoaderParameters(BaseParameters):
    channel_ids: List[str] = []
    earliest_date: Optional[datetime] = datetime(2023, 3, 10, 0, 0)


@step(enable_cache=True)
def slack_loader(params: SlackLoaderParameters) -> List[Document]:
    # slack loader; returns langchain documents
    # SlackReader = download_loader("SlackReader")
    loader = SlackReader(
        slack_token=os.environ["SLACK_BOT_TOKEN"],
        earliest_date=params.earliest_date,
    )
    documents = loader.load_data(channel_ids=params.channel_ids)
    return [d.to_langchain_format() for d in documents]


@step(enable_cache=False)
def index_generator(
    documents: List[Document], slack_documents: List[Document]
) -> VectorStore:
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(slack_documents)
    documents.extend(texts)  # merges the two document lists
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


# @step
# def llama_index_generator(
#     documents: List[Document], slack_documents: List[Document]
# ) -> GPTFaissIndex:
#     documents = [LlamaDocument.from_langchain_format(d) for d in documents]
#     documents.extend(slack_documents)
#     faiss_index = faiss.IndexFlatL2(1536)
#     return GPTFaissIndex(documents, faiss_index=faiss_index)


# def run_langchain():
#     pipeline = docs_to_index_pipeline(
#         document_loader=docs_loader(),
#         index_generator=index_generator(),
#     )
#     pipeline.configure(enable_cache=False)
#     pipeline.run()


# def run_llama():
#     pipeline = docs_to_index_pipeline(
#         document_loader=docs_loader(),
#         slack_loader=slack_loader(),
#         index_generator=llama_index_generator(),
#     )
#     # pipeline.configure(enable_cache=False)
#     pipeline.run()


# def post_exec_llama_index():
#     integration_registry.activate_integrations()
#     pipeline = get_pipeline("docs_to_index_pipeline")
#     last_run = pipeline.runs[0]
#     index = last_run.get_step("index_generator").output.read()
#     response = index.query(
#         "Are materializers loaded automatically during the post execution workflow?"
#     )
#     print(response)


def get_zenml_versions():
    # Clone zenml repo to random dir in CWD
    random_repo_name = f"_{uuid4()}"
    repo_dir = os.path.join(os.getcwd(), random_repo_name)
    repo = git.Repo.clone_from(
        url="https://github.com/zenml-io/zenml",
        to_path=random_repo_name,
    )

    # Get all release versions
    versions = []
    for refs in repo.remote().refs:
        branch_name = refs.name
        if not branch_name.startswith("origin/release"):
            continue
        version = branch_name.split("/")[-1]
        versions.append(version)

    # Cleanup
    shutil.rmtree(repo_dir)

    return versions


def _page_exists(url: str) -> bool:
    import requests

    r = requests.get(url)
    if r.status_code == 200:
        return True
    return False


def get_release_date(package_name: str, version: str) -> datetime:
    """Get the release date of a package version.

    Args:
        package_name: Name of the package.
        version: Version of the package.
    """
    return datetime(2023, 3, 8)


def build_indices_for_zenml_versions(
    versions: List[str], pipeline_name="zenml_docs_index_generation"
):
    @pipeline(name=pipeline_name)
    def docs_to_index_pipeline(document_loader, slack_loader, index_generator):
        documents = document_loader()
        slack_docs = slack_loader()
        index_generator(documents, slack_docs)

    for version in tqdm(versions):
        base_url = "https://docs.zenml.io"
        docs_url = f"https://docs.zenml.io/v/{version}"
        if not _page_exists(docs_url):
            print(f"Couldn't find docs page for zenml version '{version}'.")
            continue
        print(f"Building index for zenml docs of version '{version}'...")
        pip = docs_to_index_pipeline(
            document_loader=docs_loader(
                params=DocsLoaderParameters(
                    docs_uri=docs_url, base_url=base_url
                )
            ),
            index_generator=index_generator(),
            slack_loader=slack_loader(
                params=SlackLoaderParameters(
                    channel_ids=SLACK_CHANNEL_IDS,
                    earliest_date=get_release_date(version),
                )
            ),
        )
        run_name = (
            f"{pipeline_name}" + "_{{{date}}}_{{{time}}}" + f"_{version}"
        )
        try:
            pip.run(run_name=run_name)
        except Exception as e:
            print(f"Failed to build index for zenml version '{version}'.")
            print(e)


def main():
    print("Fetching zenml versions...")
    # versions = get_zenml_versions()  # all release versions
    # versions = ["0.10.0", "0.35.1"]
    versions = ["0.35.1"]
    print(f"Found {len(versions)} versions.")
    print("Building indices for zenml versions...")
    build_indices_for_zenml_versions(versions)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    # run_llama()
    # post_exec_llama_index()
    main()
