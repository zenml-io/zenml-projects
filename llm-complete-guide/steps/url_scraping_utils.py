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

import re
from logging import getLogger
from typing import List

import requests
from bs4 import BeautifulSoup

logger = getLogger(__name__)


def fetch_all_sitemap_urls(sitemap_url):
    urls = []
    try:
        response = requests.get(sitemap_url)
        soup = BeautifulSoup(response.text, 'xml')

        # Check if this is a sitemap index
        sitemapTags = soup.find_all("sitemap")
        if sitemapTags:
            # This is a sitemap index
            for sitemap in sitemapTags:
                loc = sitemap.find("loc")
                if loc:
                    sub_sitemap_url = loc.text
                    # Recursively fetch URLs from sub-sitemap
                    urls.extend(fetch_all_sitemap_urls(sub_sitemap_url))
        else:
            # This is a regular sitemap
            locations = soup.find_all("loc")
            urls.extend([loc.text for loc in locations])

    except Exception as e:
        print(f"Error fetching sitemap {sitemap_url}: {str(e)}")

    return urls


def get_all_pages(base_url: str = "https://docs.zenml.io") -> List[str]:
    """
    Retrieve all pages from the ZenML documentation sitemap.

    Args:
        base_url (str): The base URL of the documentation. Defaults to "https://docs.zenml.io"

    Returns:
        List[str]: A list of all documentation page URLs.
    """
    logger.info("Fetching sitemap from docs.zenml.io...")

    cleaned_url = base_url.rstrip("/")

    # Fetch the sitemap
    sitemap_url = f"{cleaned_url}/sitemap.xml"
    urls = fetch_all_sitemap_urls(sitemap_url)

    logger.info(f"Found {len(urls)} pages in the sitemap.")
    return urls


def extract_parent_section(url: str) -> str:
    """
    Extracts the parent section from a URL.

    Args:
        url: The URL to extract the parent section from.

    Returns:
        The parent section if found, otherwise None.
    """
    match = re.search(
        r"https://docs\.zenml\.io(?:/v(?:/(?:docs|\d+\.\d+\.\d+))?)?/([^/]+)",
        url,
    )
    return match.group(1) if match else None
