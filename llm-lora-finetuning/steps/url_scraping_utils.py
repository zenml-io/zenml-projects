import asyncio
from logging import getLogger
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = getLogger(__name__)

# Rate limiting configuration
DELAY_BETWEEN_REQUESTS = 1  # Delay in seconds between each request

# Caching configuration
visited_links = set()


async def is_valid_url(url: str, base: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.netloc) and parsed.netloc == base


async def get_all_links(session: aiohttp.ClientSession, url: str, base: str, max_retries: int = 3) -> List[str]:
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                soup = BeautifulSoup(await response.text(), "html.parser")
                links = []

                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    full_url = urljoin(url, href)
                    parsed_url = urlparse(full_url)
                    cleaned_url = parsed_url._replace(fragment="").geturl()
                    if await is_valid_url(cleaned_url, base) and cleaned_url not in visited_links:
                        links.append(cleaned_url)
                        visited_links.add(cleaned_url)
                        print(f"Found link: {cleaned_url}")

                print(f"Total links found for {url}: {len(links)}")
                return links
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout occurred for URL: {url}. Retrying (attempt {attempt + 1})...")
            else:
                logger.error(f"Max retries reached for URL: {url}. Skipping...")
                return []
        finally:
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)  # Delay between requests


async def crawl(session: aiohttp.ClientSession, url: str, base: str, max_depth: int, current_depth: int = 0) -> None:
    if current_depth >= max_depth or url in visited_links:
        return

    visited_links.add(url)
    print(f"Crawling URL: {url}")
    links = await get_all_links(session, url, base)

    tasks = []
    for link in links:
        if link not in visited_links:
            print(f"Queuing URL: {link}")
            task = asyncio.create_task(crawl(session, link, base, max_depth, current_depth + 1))
            tasks.append(task)

    await asyncio.gather(*tasks)


async def get_all_pages(url: str, max_depth: int = 3, timeout: int = 10) -> List[str]:
    logger.debug(f"Scraping all pages from {url}...")
    base_url = urlparse(url).netloc

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        await crawl(session, url, base_url, max_depth)

    logger.debug(f"Found {len(visited_links)} pages.")
    logger.debug("Done scraping pages.")
    return list(visited_links)

def get_readme_urls(repo_url: str) -> Tuple[List[str], List[str]]:
    """
    Retrieve folder and README links from a GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists: folder links and README links.
    """
    headers = {"Accept": "application/vnd.github+json"}
    r = requests.get(repo_url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    folder_links = []
    readme_links = []

    for link in soup.find_all("a", class_="js-navigation-open Link--primary"):
        href = link["href"]
        full_url = f"https://github.com{href}"
        if "tree" in href:
            folder_links.append(full_url)
        elif "README.md" in href:
            readme_links.append(full_url)

    return folder_links, readme_links


def get_nested_readme_urls(repo_url: str) -> List[str]:
    """
    Retrieve all nested README links from a GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.

    Returns:
        List[str]: A list of all nested README links.
    """
    folder_links, readme_links = get_readme_urls(repo_url)

    for folder_link in folder_links:
        _, nested_readme_links = get_readme_urls(folder_link)
        readme_links.extend(nested_readme_links)

    return readme_links
