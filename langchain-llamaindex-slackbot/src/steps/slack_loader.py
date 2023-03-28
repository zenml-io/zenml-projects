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

import datetime
import os
from typing import List, Optional

from langchain.docstore.document import Document
from llama_index.readers.slack import SlackReader
from slack_sdk import WebClient
from zenml.steps import BaseParameters, step


def get_channel_id_from_name(name: str) -> str:
    """Gets a channel ID from a Slack channel name.

    Args:
        name: Name of the channel.

    Returns:
        Channel ID.
    """
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    response = client.conversations_list()
    conversations = response["channels"]
    if id := [c["id"] for c in conversations if c["name"] == name][0]:
        return id
    else:
        raise ValueError(f"Channel {name} not found.")


SLACK_CHANNEL_IDS = [get_channel_id_from_name("general")]


class SlackLoaderParameters(BaseParameters):
    channel_ids: List[str] = []
    earliest_date: Optional[datetime.datetime] = None
    latest_date: Optional[datetime.datetime] = None


@step(enable_cache=True)
def slack_loader(params: SlackLoaderParameters) -> List[Document]:
    # slack loader; returns langchain documents
    # SlackReader = download_loader("SlackReader")
    loader = SlackReader(
        slack_token=os.environ["SLACK_BOT_TOKEN"],
        earliest_date=params.earliest_date,
        latest_date=params.latest_date,
    )
    documents = loader.load_data(channel_ids=params.channel_ids)
    return [d.to_langchain_format() for d in documents]
