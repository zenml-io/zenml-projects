from zenml import pipeline

from steps.download_urls import url_scraper, web_url_loader


@pipeline
def docs_loader_pipeline():
    """Generate index for ZenML."""
    urls = url_scraper()
    documents = web_url_loader(urls)
