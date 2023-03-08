import logging
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import GitbookLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from llama_index import download_loader
from zenml.pipelines import pipeline
from zenml.steps import BaseParameters, step


@pipeline
def docs_to_index_pipeline(
    document_loader, embedding_model_loader, index_generator
):
    documents = document_loader()
    embeddings = embedding_model_loader()
    index_generator(documents, embeddings)


class IndexGeneratorParameters(BaseParameters):

    docs_uri: str = "https://docs.zenml.io"


@step
def docs_loader(params: IndexGeneratorParameters) -> List[Document]:
    # loader = GitbookLoader(params.docs_uri)
    # page_data = loader.load()
    loader = GitbookLoader(params.docs_uri, load_all_paths=True)
    all_pages_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(all_pages_data)
    return documents


class SlackLoaderParameters(BaseParameters):

    channel_ids: List[str] = []


@step
def slack_loader(params: SlackLoaderParameters) -> List[Document]:

    SlackReader = download_loader("SlackReader")
    loader = SlackReader()
    documents = loader.load_data(channel_ids=params.channel_ids)
    return documents


@step
def embedding_model_loader() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


@step
def index_generator(
    documents: List[Document], embeddings: OpenAIEmbeddings
) -> VectorStore:
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


# def run_chat_bot():
#     qa_chain = get_chain(vectorstore, question_handler, stream_handler)


def main():
    pipeline = docs_to_index_pipeline(
        document_loader=docs_loader(),
        embedding_model_loader=embedding_model_loader(),
        index_generator=index_generator(),
    )
    # pipeline.configure(enable_cache=False)
    pipeline.run()


def post_exec_llama_index():
    import faiss
    from llama_index import Document as LlamaDocument
    from llama_index import GPTFaissIndex
    from zenml.integrations.registry import integration_registry
    from zenml.post_execution import get_pipeline

    integration_registry.activate_integrations()

    pipeline = get_pipeline("docs_to_index_pipeline")
    runs = pipeline.runs
    run = runs[0]  # TODO: -1 should be the latest run
    documents = run.get_step("document_loader").output.read()
    documents = [LlamaDocument.from_langchain_format(d) for d in documents]

    # Creating a faiss index
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)

    # Load documents, build the GPTFaissIndex
    index = GPTFaissIndex(documents, faiss_index=faiss_index)
    response = index.query(
        "Are materializers loaded automatically during the post execution workflow?"
    )
    print(response)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    main()
