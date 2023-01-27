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
from typing import List

import bbc_feeds
import requests
from zenml.steps import step, BaseParameters
from bs4 import BeautifulSoup


class BBCParameters(BaseParameters):
    """"""
    news_limit = 10

    pass


def download(link):
    return requests.get(link)


@step
def bbc_news_source(params: BBCParameters) -> List[str]:
    """"""

    for story in bbc_feeds.news().top_stories(limit=params.news_limit):

        content = download(story.link).content

        content = BeautifulSoup(content, 'html.parser')
        content_body = content.findAll('body')
        paragraphs = content_body[0].findAll('p')

        extracted_paragraphs = {}
        for p in paragraphs:
            print(p)
            class_identifier = "-".join(p['class'])
            if "PromoHeadline" not in class_identifier:
                if class_identifier not in extracted_paragraphs:
                    extracted_paragraphs[class_identifier] = {
                        "len": 0,
                        "text": ""
                    }

                extracted_paragraphs[class_identifier]["len"] += len(p.text)
                extracted_paragraphs[class_identifier]["text"] += p.text

        article_text = extracted_paragraphs[
            max(
                extracted_paragraphs,
                key=lambda x: extracted_paragraphs[x]["len"]
        )]["text"]

    return [article_text]
