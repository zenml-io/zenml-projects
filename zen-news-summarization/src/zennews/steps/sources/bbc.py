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
from typing import List, Dict, Any

import bbc_feeds
import requests
from bs4 import BeautifulSoup
from zenml.steps import step, BaseParameters

from zennews.models.article import Article

EXCLUDED_CLASS_IDENTIFIERS = ["PromoHeadline"]


class BBCParameters(BaseParameters):
    """Parameters to modify your feed from BBC news."""
    top_stories: bool = True
    uk: bool = False
    tech: bool = False
    business: bool = False
    entertainment: bool = False
    north_america: bool = False
    science: bool = False
    world: bool = False

    limit_per_section: int = 5

    def gather_feed(self) -> Dict[Any, Any]:
        """Method to iterate through parameters to gather the news feed.

        Return:
            dict, keys denote the categories, values denote the actual feed
        """
        news = getattr(bbc_feeds, "news")()
        feed = {}
        for k, v in self.dict().items():
            if k != 'limit_per_section':
                if v:
                    subsection_method = getattr(news, k)
                    feed[k] = subsection_method(limit=self.limit_per_section)
        return feed


@step
def bbc_news_source(params: BBCParameters) -> List[Article]:
    """Step to download and parse news articles from BBC News."""
    articles: List[Article] = []

    # Derive the feed from the given params
    feed = params.gather_feed()

    # The sections include all the categories in the params such as "news_X"
    for section_name, section_feed in feed.items():
        # Each section features its own RSS feed
        for story in section_feed:
            # Download the content from the story
            content = requests.get(story.link).content

            # Parse out all the paragraphs
            content = BeautifulSoup(content, 'html.parser')
            content_body = content.findAll('body')
            all_paragraphs = content_body[0].findAll('p')

            # Find out the longest paragraph in the content
            paragraphs = {}
            for p in all_paragraphs:
                if p.get("class", None):
                    class_identifier = "-".join(p['class'])
                    if any(
                        [h not in class_identifier for h in
                         EXCLUDED_CLASS_IDENTIFIERS]
                        ):
                        if class_identifier not in paragraphs:
                            paragraphs[class_identifier] = {
                                "len": 0,
                                "text": ""
                            }

                        paragraphs[class_identifier]["len"] += len(p.text)
                        paragraphs[class_identifier]["text"] += p.text

            article_text = paragraphs[
                max(
                    paragraphs,
                    key=lambda x: paragraphs[x]["len"]
                )]["text"]

            # Create an "Article" from the result and add it to the list
            articles.append(
                Article(
                    source="bbc",
                    section=f"{section_name}",
                    url=str(story.link),
                    text=article_text,
                )
            )

    return articles
