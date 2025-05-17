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


from typing import Any, Callable

from zenml.pipelines import pipeline

pipeline_name = "daily_supabase_summary"


@pipeline(name=pipeline_name)
def daily_supabase_summary(
    get_latest_data: Callable[[], Any],
    generate_summary: Callable[[Any], Any],
    report_summary: Callable[[Any], Any],
) -> None:
    """Generates a summary of the latest data.

    Args:
        get_latest_data (step): Get the latest data from Supabase.
        generate_summary (step): Generate a summary of the data.
        report_summary (step): Report the summary.
    """
    data = get_latest_data()
    summary = generate_summary(data)
    report_summary(summary)
