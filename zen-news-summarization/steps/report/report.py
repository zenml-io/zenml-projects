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

from models import Article


def generate_final_report(summaries: List[Article]):
    # TODO: Improve the final report
    return "".join([s.text for s in summaries])


@step
def post_summaries(summaries: List[Article]) -> str:
    """Step that reports the summaries through an alerter (if registered)"""
    final_report = generate_final_report(summaries=summaries)

    # Fetch the alerter if defined and use it to send the final report
    client = Client()
    if client.active_stack.alerter:
        client.active_stack.alerter.post(final_report)
    else:
        logging.warning(
            'There is no alerter defined in your stack. The result will still'
            'be saved as an artifact in your artifact store.'
        )

    return final_report
