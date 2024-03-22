from typing import Any, List

from langchain_community.document_loaders import UnstructuredURLLoader
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step

from steps.url_scraping_utils import get_all_pages


@step(enable_cache=True)
def url_scraper(
    docs_url: str = "https://docs.zenml.io",
    repo_url: str = "https://github.com/zenml-io/zenml",
    website_url: str = "https://zenml.io",
) -> Annotated[List[str], "urls"]:
    """Generates a list of relevant URLs to scrape.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        release_notes_url: URL to the release notes.
        website_url: URL to the website.

    Returns:
        List of URLs to scrape.
    """
    # # We comment this out to make this pipeline faster
    # # examples_readme_urls = get_nested_readme_urls(repo_url)
    # docs_urls = get_all_pages(docs_url)
    # # website_urls = get_all_pages(website_url)
    # # all_urls = docs_urls + website_urls + examples_readme_urls
    # all_urls = docs_urls
    # log_artifact_metadata(
    #     metadata={
    #         "count": len(all_urls),
    #     },
    # )
    # return all_urls
    #TODO revert this once testing is finished
    return ["https://docs.zenml.io/", "https://zenml.io/"]


@step(enable_cache=True)
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
