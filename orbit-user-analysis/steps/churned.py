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
from datetime import datetime, timedelta, timezone

from constants import CHURNED_TAG
from dateutil import parser
from steps.utils import list_members, update_member_tags
from zenml import step


@step
def churned(
    check_days: int = 60,
    inactive_days: int = 14,
) -> None:
    """Step that detects churned users and tags them accordingly.

    Args:
        check_days: the number of days to check in the past
        inactive_days: the number of inactive days required before tagging
            someone churned.

    Returns:
        json string representing the list of user and their metadata
    """

    # The first section is about the clean-up
    existing_churned_members = list_members(tags=CHURNED_TAG)

    for member in existing_churned_members:
        last_activity = member["attributes"]["last_activity_occurred_at"]
        last_activity_t = parser.isoparse(last_activity)
        last_activity_delta = datetime.now(timezone.utc) - last_activity_t

        if last_activity_delta < timedelta(days=inactive_days):
            tags = member["attributes"]["tags"] or []

            if CHURNED_TAG in tags:
                tags.remove(CHURNED_TAG)

                member_slug = member["attributes"]["slug"]
                update_member_tags(member_slug, tags)

    # The second section is about detection new churned users
    recent_members = list_members(days=check_days)

    for member in recent_members:
        last_activity = member["attributes"]["last_activity_occurred_at"]
        last_activity_t = parser.isoparse(last_activity)
        last_activity_delta = datetime.now(timezone.utc) - last_activity_t

        if last_activity_delta > timedelta(days=inactive_days):
            tags = member["attributes"]["tags"] or []

            if CHURNED_TAG not in tags:
                tags.append(CHURNED_TAG)

                member_slug = member["attributes"]["slug"]
                update_member_tags(member_slug, tags)
