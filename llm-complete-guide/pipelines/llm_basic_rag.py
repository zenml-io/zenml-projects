from zenml import pipeline
from steps.url_scraper import url_scraper
from steps.web_url_loader import web_url_loader


@pipeline
def llm_basic_rag() -> None:
    """Pipeline to train a basic RAG model."""

    urls = url_scraper()
    web_url_loader(urls=urls)
