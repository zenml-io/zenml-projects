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
import logging
from typing import List

from zenml.client import Client
from zenml.steps import step

from zennews.models import Article

from mdutils import MdUtils
from datetime import datetime


def generate_final_report(articles: List[Article]):
    md_file = MdUtils(file_name='report', title='ZenNews Summaries')
    md_file.new_header(
        title=f'From {articles[0].source.upper()} generated at '
              f'{datetime.now().strftime("%m/%d/%Y %H:%M:%S")}',
        level=1,
    )

    for a in articles:
        md_file.new_paragraph(f"**[{a.section}]** {a.text} [Link]({a.url})")

    return md_file.file_data_text


@step
def post_summaries(articles: List[Article]) -> str:
    """Step that reports the summaries through an alerter (if registered)"""
    final_report = generate_final_report(articles=articles)

    # Fetch the alerter if defined and use it to send the final report
    client = Client()
    if client.active_stack.alerter:
        client.active_stack.alerter.post(message=articles, params=None) # noqa
    else:
        logging.warning(
            'There is no alerter defined in your stack. The result will still'
            'be saved as an artifact in your artifact store.'
        )

    return final_report
