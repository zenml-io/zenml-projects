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

from zenml.steps import step
from zenml.client import Client


from typing import List


@step
def post_summaries(summaries: List[str]) -> None:
    """"""
    client = Client()

    message = 'aasdf'
    if client.active_stack.alerter:
        client.active_stack.alerter.post(message)
    else:
        logging.warning('YO yo no alerter')
        raise RuntimeError(
            "Step `slack_alerter_post_step` requires an alerter component of "
            "flavor `slack`, but the currently active alerter is of type "
            f", which is not a subclass of `SlackAlerter`."
        )