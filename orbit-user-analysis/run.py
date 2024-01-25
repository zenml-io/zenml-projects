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


import argparse

from pipelines import community_analysis_pipeline
from zenml.config.schedule import Schedule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-w", "--weekly", action="store_true")
    group.add_argument("-d", "--daily", action="store_true")

    args = parser.parse_args()

    if args.daily:
        schedule = Schedule(cron_expression="0 9 * * *")
    elif args.weekly:
        schedule = Schedule(cron_expression="0 9 * * MON")
    else:
        schedule = None

    community_analysis_pipeline = community_analysis_pipeline.with_options(
        schedule=schedule
    )
    community_analysis_pipeline()
