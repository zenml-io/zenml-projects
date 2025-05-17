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

import logging

from pipelines.supabase_summary import daily_supabase_summary
from steps.alerters import print_alerter
from steps.importers import supabase_reader
from steps.summarizers import gpt_4_summarizer
from zenml.client import Client


def main() -> None:
    if Client().active_stack.alerter is None:
        # we use a print alerter
        alerter = print_alerter()
    else:
        # We assume it's a slack alerter
        from zenml.integrations.slack.steps.slack_alerter_post_step import (
            slack_alerter_post_step,
        )

        alerter = slack_alerter_post_step()

    daily_supabase_summary(
        get_latest_data=supabase_reader(),
        generate_summary=gpt_4_summarizer(),
        report_summary=alerter,
    ).run()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    logging.getLogger().setLevel(logging.INFO)
    main()
