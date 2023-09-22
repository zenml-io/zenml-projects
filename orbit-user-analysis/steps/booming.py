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


from constants import BOOMING_TAG
from steps.utils import list_members, update_member_tags
from zenml import step


@step
def booming(check_days: int = 14, booming_threshold: int = 150) -> None:
    """Step that detects users with activity above a certain threshold.

    Args:
        check_days: the number of days to check in the past
        booming_threshold: the minimum number of events someone has to conduct
            to be tagged with the booming tag.

     Returns:
         json string representing the list of users and their metadata
    """

    # The first section is about the clean-up
    existing_booming_members = list_members(tags=BOOMING_TAG)

    for member in existing_booming_members:
        tags = member["attributes"]["tags"] or []

        if BOOMING_TAG in tags:
            tags.remove(BOOMING_TAG)

            member_slug = member["attributes"]["slug"]
            update_member_tags(member_slug, tags)

    # The second section is about detection new booming users
    new_booming_members = list_members(
        days=check_days,
        activities_count_min=booming_threshold,
    )

    for member in new_booming_members:
        tags = member["attributes"]["tags"] or []

        if BOOMING_TAG not in tags:
            tags.append(BOOMING_TAG)

            member_slug = member["attributes"]["slug"]
            update_member_tags(member_slug, tags)
