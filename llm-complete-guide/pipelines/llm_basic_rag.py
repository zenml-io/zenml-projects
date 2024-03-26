from steps.populate_index import (
    generate_embeddings,
    index_generator,
    preprocess_documents,
)
from steps.url_scraper import url_scraper
from steps.web_url_loader import web_url_loader
from zenml import pipeline


@pipeline
def llm_basic_rag() -> None:
    """Pipeline to train a basic RAG model."""

    urls = url_scraper()
    docs = web_url_loader(urls=urls)
    processed_docs = preprocess_documents(documents=docs)
    embeddings = generate_embeddings(split_documents=processed_docs)
    index_generator(embeddings=embeddings, documents=docs)
