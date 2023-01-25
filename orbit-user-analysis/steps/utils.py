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


from zenml.client import Client
import requests


def list_users(days: int = None, tags: str = None, **kwargs):
    """Function to list all members within the number of specified days.

    Attributes:
        days: int, the last number of days to take into consideration
        tags: str, the tags to filter with separated by a ','

    Returns:
        the list of users
    """
    # Start with an empty user list
    users = []

    # Set the params
    query_args = {
        "affiliation": "member",
        "items": "100",
    }
    query_args.update(kwargs)

    # Create the header and the URL
    secret_manager = Client().active_stack.secrets_manager

    workspace = secret_manager.get_secret("orbit").content["ORBIT_WORKSPACE"]
    token = secret_manager.get_secret("orbit").content["ORBIT_API_TOKEN"]

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {token}"
    }

    url = f"https://app.orbit.love/api/v1/{workspace}/members?" \
          f"{'&'.join({f'{k}={v}' for k, v in query_args.items()})}" \
          f"{f'&relative=this_{days}_days' if days is not None else ''}" \
          f"{f'&member_tags={tags}' if tags is not None else ''}"

    # Go through the pages and add to the list
    page = requests.get(url, headers=headers).json()
    while True:
        users.extend(page['data'])
        if page["links"]["next"] is not None:
            page = requests.get(page["links"]["next"], headers=headers).json()
        else:
            break
    return users
