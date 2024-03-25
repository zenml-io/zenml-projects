import asyncio
from typing import Any, List

from langchain_community.document_loaders import UnstructuredURLLoader
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step

from steps.url_scraping_utils import get_all_pages


@step(enable_cache=False)
def url_scraper(
    docs_url: str = "https://docs.zenml.io",
    repo_url: str = "https://github.com/zenml-io/zenml",
    website_url: str = "https://zenml.io",
    max_depth: int = 3,
) -> Annotated[List[str], "urls"]:
    """Generates a list of relevant URLs to scrape.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        website_url: URL to the website.
        max_depth: Maximum depth for crawling.

    Returns:
        List of URLs to scrape.
    """
    docs_urls = asyncio.run(get_all_pages(docs_url, max_depth))
    all_urls = docs_urls
    log_artifact_metadata(
        metadata={
            "count": len(all_urls),
        },
    )
    return all_urls


@step(enable_cache=False)
def web_url_loader(urls: List[str]) -> List[Any]:
    """Loads documents from a list of URLs.

    Args:
        urls: List of URLs to load documents from.

    Returns:
        List of langchain documents.
    """
    loader = UnstructuredURLLoader(
        urls=urls,
    )
    return loader.load()
