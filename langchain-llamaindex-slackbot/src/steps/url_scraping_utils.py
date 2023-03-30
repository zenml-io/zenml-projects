#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = getLogger(__name__)


def is_valid_url(url, base):
    parsed = urlparse(url)
    return bool(parsed.netloc) and parsed.netloc == base


def get_all_links(url, base):
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


def crawl(url, base, visited=None):
    if visited is None:
        visited = set()

    visited.add(url)
    links = get_all_links(url, base)

    for link in links:
        if link not in visited:
            visited.update(crawl(link, base, visited))

    return visited


def get_all_pages(url):
    logger.debug(f"Scraping all pages from {url}...")
    base_url = urlparse(url).netloc
    pages = crawl(url, base_url)
    logger.debug(f"Found {len(pages)} pages.")
    logger.debug("Done scraping pages.")
    return list(pages)


def get_readme_urls(repo_url):
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
            # raw_url = full_url.replace(
            #     "github.com", "raw.githubusercontent.com"
            # ).replace("/blob/", "/")
            readme_links.append(full_url)

    return folder_links, readme_links


def get_nested_readme_urls(repo_url):
    folder_links, readme_links = get_readme_urls(repo_url)

    for folder_link in folder_links:
        _, nested_readme_links = get_readme_urls(folder_link)
        readme_links.extend(nested_readme_links)

    return readme_links
