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
    """Executes the pipeline to train a basic RAG model.

    This function performs the following steps:
    1. Scrapes URLs using the url_scraper function.
    2. Loads documents from the scraped URLs using the web_url_loader function.
    3. Preprocesses the loaded documents using the preprocess_documents function.
    4. Generates embeddings for the preprocessed documents using the generate_embeddings function.
    5. Generates an index for the embeddings and documents using the index_generator function.
    """
    urls = url_scraper()
    docs = web_url_loader(urls=urls)
    processed_docs = preprocess_documents(documents=docs)
    embeddings = generate_embeddings(split_documents=processed_docs)
    index_generator(embeddings=embeddings, documents=docs)
