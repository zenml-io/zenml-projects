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

from zenml.steps import step, BaseParameters

from steps.utils import list_users


class BoomingParameters(BaseParameters):
    check_days: int = 30
    booming_threshold: int = 150


@step
def booming(params: BoomingParameters) -> None:
    """ """

    # The first section is about the clean-up
    booming_users = list_users(tags='booming')

    for user in booming_users:
        # TODO: Remove tag
        print(f'Recently booming user: {user["attributes"]["name"]}')
        print("-----")

    # The second section is about detection new booming users
    users = list_users(
        days=params.check_days,
        activities_count_min=params.booming_threshold,
    )

    for user in users:
        # TODO: Add tag
        print(f'Booming user: {user["attributes"]["name"]}')
        print("-----")
