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

# TODO: Error handling

from typing import List

import click

from cli.base import cli
from cli.constants import SUPPORTED_SOURCES
from cli.utils import (
    parse_args,
    save_profile,
    load_profiles,
    delete_profile,
    display_profiles
)
from models.profile import Profile


@cli.group('profile')
def profile() -> None:
    """Base group for ZenNews profiles."""


@profile.command('create', context_settings={"ignore_unknown_options": True})
@click.option(
    "--name",
    "-n",
    required=True,
    help="The name of the profile"
)
@click.option(
    "--source",
    "-s",
    required=True,
    help="The name of the news source."
)
@click.option(
    "--stack",
    "-z",
    type=str,
    help="The name of the ZenML stack to run the pipeline on."
)
@click.option(
    "--frequency",
    "-f",
    type=str,
    help="The frequency of the schedule, i.e. if daily -> '1d'."
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def create_profile(
    name: str,
    source: str,
    stack: str = None,
    frequency: str = None,
    args: List[str] = None,
) -> None:
    """Creates a ZenNews profile.

    ARGS are the additional parameters to configure the news source. (Each
    arg needs to be provided with either the '--key value' format or
    the '--key' format, in which case the value is True).

    For instance:

    '--name' can be 'my_profile', '--source' can be 'bbc' and the ARGS can
    include 'news_tech' and 'sports_tennis'. In this case, the command needs
    to be structured as follows:

    'zennews create profile my_profile bbc --news_tech --sports_tennis'

    """
    # Parse the arguments
    parsed_args = parse_args(args=args)

    source_args = SUPPORTED_SOURCES.get(source)(**parsed_args)

    save_profile(
        Profile(
            name=name,
            source=source,
            args=source_args,
            stack=stack,
            frequency=frequency,
        )
    )


@profile.command('remove')
@click.argument("name")
def remove_profile(name) -> None:
    """Removes a ZenNews profile."""
    # TODO: Check whether the profile is active first
    delete_profile(name=name)


@profile.command('list')
def list_profiles() -> None:
    """Lists all created ZenNews profiles."""
    profiles = load_profiles()
    display_profiles(profiles)


@profile.command("update")
@click.argument("name")
@click.option(
    "--new_name",
    "-n",
    type=str,
    help="The name of the profile"
)
@click.option(
    "--source",
    "-s",
    type=str,
    help="The name of the news source."
)
@click.option(
    "--stack",
    "-z",
    type=str,
    help="The name of the ZenML stack to run the pipeline on."
)
@click.option(
    "--frequency",
    "-f",
    type=str,
    help="The frequency of the schedule, ie if daily -> '1d'."
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def update_profile(
    name,
    new_name,
    source,
    stack,
    frequency,
    args,
) -> None:
    """Updates a ZenNews profile."""



@profile.command("activate")
@click.argument("name")
def activate_profile(name) -> None:
    """Activates a ZenNews profile."""
    # TODO: Check if stack is present and validated
    # TODO: Check whether stack has an orchestrator with schedule support
    # TODO: Check whether stack has an alerter
    # TODO: Check whether frequency is correctly formatted.


@profile.command("deactivate")
@click.argument("name")
def deactivate_profile(name) -> None:
    """Removes a ZenNews profile."""
