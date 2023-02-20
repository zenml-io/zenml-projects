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

import json

from zenml.steps import step, BaseParameters

from constants import BOOMING_TAG
from steps.utils import list_members, update_member_tags


class BoomingParameters(BaseParameters):
    """Parameters for the 'booming' step. 
    
    Attributes:
        check_days: the number of days     
        
    """
    check_days: int = 14
    booming_threshold: int = 150


@step
def booming(params: BoomingParameters) -> str:
    """Step that detects users with activity above a certain threshold.

     Args:
         params: parameters for the step

     Returns:
         json string representing the list of users and their metadata
     """

    # The first section is about the clean-up
    existing_booming_members = list_members(
        tags=BOOMING_TAG
    )

    for member in existing_booming_members:
        tags = member["attributes"]["tags"] or []

        if BOOMING_TAG in tags:
            tags.remove(BOOMING_TAG)

            member_slug = member["attributes"]["slug"]
            update_member_tags(member_slug, tags)

    # The second section is about detection new booming users
    new_booming_members = list_members(
        days=params.check_days,
        activities_count_min=params.booming_threshold,
    )

    for member in new_booming_members:
        tags = member["attributes"]["tags"] or []

        if BOOMING_TAG not in tags:
            tags.append(BOOMING_TAG)

            member_slug = member["attributes"]["slug"]
            update_member_tags(member_slug, tags)

    return json.dumps(new_booming_members)
