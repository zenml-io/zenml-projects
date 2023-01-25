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
from datetime import datetime, timezone, timedelta
from dateutil import parser
from zenml.steps import step, BaseParameters
from pydantic import validator

from steps.utils import list_users


class ChurnedParameters(BaseParameters):
    check_days: int = 30
    inactive_days: int = 7

    @validator('inactive_days', always=True)
    def inactive_days_must_be_smaller(cls, v, values):
        """ """
        if v > values["check_days"]:
            raise ValueError(
                'The value for inactive days must be smaller than the value '
                'for check days'
            )
        return v


@step
def churned(params: ChurnedParameters) -> None:
    """ """

    # The first section is about the clean-up
    churned_users = list_users(tags='churned')

    for user in churned_users:
        last_activity = user["attributes"]["last_activity_occurred_at"]
        last_activity_t = parser.isoparse(last_activity)
        last_activity_delta = datetime.now(timezone.utc) - last_activity_t

        if last_activity_delta > timedelta(days=params.inactive_days):
            # TODO: Remove tag
            print('Churned user came back.')
            print("-----")

    # The second section is about detection new churned users
    users = list_users(days=params.check_days)

    for user in users:
        last_activity = user["attributes"]["last_activity_occurred_at"]
        last_activity_t = parser.isoparse(last_activity)
        last_activity_delta = datetime.now(timezone.utc) - last_activity_t

        if last_activity_delta > timedelta(days=params.inactive_days):
            # TODO: Add tag
            print(f'Churned user: {user["attributes"]["name"]}')
            print("-----")
