from zenml import pipeline
from steps.url_scraper import url_scraper
from steps.web_url_loader import web_url_loader
from steps.populate_index import (
    preprocess_documents,
    generate_embeddings,
    index_generator,
)


@pipeline
def llm_basic_rag() -> None:
    """Pipeline to train a basic RAG model."""

    urls = url_scraper()
    docs = web_url_loader(urls=urls)
    processed_docs = preprocess_documents(documents=docs)
    embeddings = generate_embeddings(split_documents=processed_docs)
    index_generator(embeddings=embeddings, documents=docs)
