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

from typing import Any, Dict, List, Tuple

import requests
from zenml.client import Client


def get_discord_secret() -> str:
    """Function to get the Discord secret to retrieve the webhook url."""
    webhook_url = Client().get_secret("discord").secret_values["webhook_url"]

    return webhook_url


def get_orbit_secrets() -> Tuple[str, str]:
    """Function to get the Orbit secret to retrieve the workspace and token.

    Returns:
        tuple, workspace name and API token
    """
    client = Client()

    workspace = client.get_secret("orbit").secret_values["workspace"]
    token = client.get_secret("orbit").secret_values["api_token"]

    return workspace, token


def list_members(days: int = None, tags: str = None, **kwargs) -> List[Dict[Any, Any]]:
    """Function to list all members within the number of specified days.

    Args:
        days: int, the last number of days to take into consideration
        tags: str, the tags to filter with separated by a ','

    Returns:
        the list of users
    """
    workspace, token = get_orbit_secrets()

    # Set the params
    query_args = {"affiliation": "member", "items": "100"}
    query_args.update(kwargs)

    # Create the headers and the URL
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {token}",
    }

    url = (
        f"https://app.orbit.love/api/v1/{workspace}/members?"
        f"{'&'.join({f'{k}={v}' for k, v in query_args.items()})}"
        f"{f'&relative=this_{days}_days' if days is not None else ''}"
        f"{f'&member_tags={tags}' if tags is not None else ''}"
    )

    # Go through the pages and add to the list
    page = requests.get(url, headers=headers).json()

    users = []
    while True:
        users.extend(page["data"])
        if page["links"]["next"] is not None:
            page = requests.get(page["links"]["next"], headers=headers).json()
        else:
            break

    return users


def update_member_tags(member: str, tags: List[str]) -> None:
    """Function to update a user.

    Args:
        member: str, the member slug
        tags: list of new tags for the user
    """
    workspace, token = get_orbit_secrets()

    # Create the headers and the url
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
    }

    url = f"https://app.orbit.love/api/v1/{workspace}/members/{member}"

    # Create the payload
    payload = {"tags": ", ".join(tags)}

    requests.put(url, json=payload, headers=headers)
