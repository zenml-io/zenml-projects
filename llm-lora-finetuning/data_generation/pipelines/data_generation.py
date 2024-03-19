from zenml import pipeline

from steps.download_urls import url_scraper, web_url_loader


@pipeline
def data_generation_pipeline():
    """Generate index for ZenML.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        release_notes_url: URL to the release notes.
        website_url: URL to the website.
    """
    urls = url_scraper()
    documents = web_url_loader(urls)
    # vector_store = index_generator(documents)
    # _ = agent_creator(vector_store=vector_store)
