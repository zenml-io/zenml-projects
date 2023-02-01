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
import os
from datetime import datetime

import click

from cli.constants import APP_NAME, PROFILES_PATH, CONFIG_PATH
from cli.utils import (
    save_config,
    load_config,
    load_profile,
    error,
    warning,
    stack_handler,
    display_articles, load_profiles,
)
from models import Config
from zenml.client import Client
from zenml.enums import SorterOps, GenericFilterOps
from zenml.post_execution import PipelineRunView


@click.group(APP_NAME, invoke_without_command=True)
@click.option(
    "--tests_included",
    "-t",
    is_flag=True,
    default=False,
    help="Flag to decide if test results should be displayed."
)
@click.option(
    '--name',
    '-n',
    type=str,
)
@click.pass_context
def cli(ctx, tests_included: bool, name: str):
    """CLI base command for ZenML."""

    # Supress warning messages during client initializations
    os.environ["ZENML_ENABLE_REPO_INIT_WARNINGS"] = "false"

    # Create profiles directory if it does not exist
    if not os.path.exists(PROFILES_PATH):
        os.makedirs(PROFILES_PATH)

    # Create zennews configuration file if it does not exist
    if not os.path.exists(CONFIG_PATH):
        save_config(Config())
    config = load_config()

    # Display summaries if there are no invoked sub-commands
    if not ctx.invoked_subcommand:
        click.secho(
            r"""
                     ______          _   _                   
                    |___  /         | \ | |                  
                       / / ___ _ __ |  \| | _____      _____ 
                      / / / _ \ '_ \| . ` |/ _ \ \ /\ / / __|
                     / /_|  __/ | | | |\  |  __/\ V  V /\__ \
                    /_____\___|_| |_|_| \_|\___| \_/\_/ |___/
                                                                                   
                          This is where you get the news.
             """,
            fg="magenta",
        )

        # Decide which profiles to check
        if name:
            profiles = [load_profile(name).name]
        else:
            if tests_included:
                profiles = [p.name for p in load_profiles()]
            else:
                profiles = config.active_profiles

        # Go over the profiles
        if profiles:
            client = Client()
            for profile_name in config.active_profiles:
                # The implementation of this filter is designed this way
                # due to a small bug in the 'STARTSWITH' operator. Once
                # the fix is implemented we can change it for the better.
                if tests_included:
                    name_filter = f"{profile_name}_"
                else:
                    name_filter = f"news_{profile_name}_"

                # Query the last run
                last_run = client.list_runs(
                    name=f"{GenericFilterOps.CONTAINS}:{name_filter}",
                    sort_by=f"{SorterOps.DESCENDING}:created",
                    size=1,
                ).items

                if last_run:
                    run_view = PipelineRunView(last_run[0])
                    step_view = run_view.get_step("report")
                    artifact_view = step_view.outputs['output']
                    summaries = artifact_view.read()
                    display_articles(summaries)
                else:
                    warning(
                        f"No results found for profile: {profile_name}"
                    )

        else:
            warning(
                "You do not have any active profiles at the moment. If you "
                "would like the test results to be displayed please use the "
                "'--tests-included' or '-t' flag."
            )


@cli.command("test")
@click.argument("name", type=str)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    type=bool,
    help="Skip the confirmation."
)
def test(name: str, force: bool) -> None:
    """CLI command to test the news summarization on the default stack.

    Args:
        name: str, the name of a profile.
        force: bool, flag to skip the confirmation prompt.
    """
    if not force:
        warning(
            'Testing a profile will temporarily change your active stack to '
            'the default stack and run the pipeline locally. This also means '
            'that the pipeline will download and utilize the model locally.'
        )
        if not click.confirm('Would you like to continue?'):
            return None

    from pipelines import zen_news_pipeline
    from steps import bart_large_cnn_samsum, post_summaries, SOURCE_STEP_MAPPING

    if name:
        profile_names = [name]
    else:
        config = load_config()
        if not config.active_profiles:
            error(
                "There are no active profiles. Either set a profile active"
                "or provide a profile name."
            )
        profile_names = config.active_profiles

    for profile_name in profile_names:
        profile = load_profile(profile_name)
        try:
            source_dict = SOURCE_STEP_MAPPING[profile.source]
            step_class = source_dict["step"]
            parameters_class = source_dict["parameters"]

        except KeyError:
            error(f"There are no sources with the name '{profile.source}'.")
            return None  # noqa

        with stack_handler():
            pipeline = zen_news_pipeline(
                collect=step_class(parameters_class(**profile.args)),
                summarize=bart_large_cnn_samsum(),
                report=post_summaries(),
            )

            run_name = f'test_{profile_name}' \
                       f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}'

            pipeline.run(run_name=run_name)


@click.command("clean")
def clean():
    """CLI command to clean the profiles folders and the config file."""
    # TODO: Implement a small clean up method.
    # TODO: Add confirmation
