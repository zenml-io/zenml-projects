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

import os
from itertools import zip_longest
from typing import List

from zenml.logger import get_logger
from zenml.post_execution import get_pipeline

logger = get_logger(__name__)


PIPELINE_NAME = os.getenv("PIPELINE_NAME", "zenml_docs_index_generation")


def connect_to_zenml_server():
    from zenml.config.global_config import GlobalConfiguration
    from zenml.exceptions import IllegalOperationError
    from zenml.zen_stores.base_zen_store import BaseZenStore

    zenml_server_url = os.getenv("ZENML_SERVER_URL")
    zenml_username = os.getenv("ZENML_USERNAME")
    zenml_password = os.getenv("ZENML_PASSWORD")

    store_dict = {
        "url": zenml_server_url,
        "username": zenml_username,
        "password": zenml_password,
    }

    store_type = BaseZenStore.get_store_type(zenml_server_url)
    store_config_class = BaseZenStore.get_store_config_class(store_type)
    assert store_config_class is not None

    store_config = store_config_class.parse_obj(store_dict)
    try:
        GlobalConfiguration().set_store(store_config)
    except IllegalOperationError as e:
        logger.warning(
            f"User '{zenml_username}' does not have sufficient permissions to "
            f"to access the server at '{zenml_server_url}'. Please ask the server "
            f"administrator to assign a role with permissions to your "
            f"username: {str(e)}"
        )


def get_vector_store():
    """Get the vector store from the pipeline."""
    pipeline = get_pipeline(pipeline=PIPELINE_NAME, version=9)
    our_run = pipeline.runs[0]
    print("Using pipeline: ", pipeline.model.name, "v", pipeline.model.version)
    print("Created on: ", pipeline.model.updated)
    return our_run.steps[-1].output.read()


def get_last_n_messages(full_thread: List[List[str]], n: int = 5):
    """Get the last n messages from a thread.

    Args:
        full_thread (list): List of messages in a thread
        n (int): Number of messages to return

    Returns:
        list: Last n messages in a thread (or the full thread if less than n)
    """
    return full_thread[-n:] if len(full_thread) >= n else full_thread


def convert_to_chat_history(messages: List[str]):
    """Convert a list of messages to a chat history.

    Args:
        messages (list): List of messages in a thread

    Returns:
        list: Chat history as a list of pairs of messages
    """
    paired_list = [
        list(filter(None, pair)) for pair in zip_longest(*[iter(messages)] * 2)
    ]
    if len(messages) % 2 == 1:
        paired_list[-1].append("")
    return paired_list
