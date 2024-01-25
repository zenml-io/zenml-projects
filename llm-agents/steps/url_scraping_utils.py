#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from logging import getLogger
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = getLogger(__name__)


def is_valid_url(url: str, base: str) -> bool:
    """
    Check if the given URL is valid and has the same base as the provided base.

    Args:
        url (str): The URL to check.
        base (str): The base URL to compare against.

    Returns:
        bool: True if the URL is valid and has the same base, False otherwise.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and parsed.netloc == base


def get_all_links(url: str, base: str) -> List[str]:
    """
    Retrieve all valid links from a given URL with the same base.

    Args:
        url (str): The URL to retrieve links from.
        base (str): The base URL to compare against.

    Returns:
        List[str]: A list of valid links with the same base.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        full_url = urljoin(url, href)
        parsed_url = urlparse(full_url)
        cleaned_url = parsed_url._replace(fragment="").geturl()
        if is_valid_url(cleaned_url, base):
            links.append(cleaned_url)

    return links


def crawl(url: str, base: str, visited: Set[str] = None) -> Set[str]:
    """
    Recursively crawl a URL and its links, retrieving all valid links with the same base.

    Args:
        url (str): The URL to crawl.
        base (str): The base URL to compare against.
        visited (Set[str]): A set of URLs that have been visited. Defaults to None.

    Returns:
        Set[str]: A set of all valid links with the same base.
    """
    if visited is None:
        visited = set()

    visited.add(url)
    links = get_all_links(url, base)

    for link in links:
        if link not in visited:
            visited.update(crawl(link, base, visited))

    return visited


def get_all_pages(url: str) -> List[str]:
    """
    Retrieve all pages with the same base as the given URL.

    Args:
        url (str): The URL to retrieve pages from.

    Returns:
        List[str]: A list of all pages with the same base.
    """
    logger.debug(f"Scraping all pages from {url}...")
    base_url = urlparse(url).netloc
    pages = crawl(url, base_url)
    logger.debug(f"Found {len(pages)} pages.")
    logger.debug("Done scraping pages.")
    return list(pages)


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
