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

from typing import List

import click

from cli.base import cli
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


@profile.command(
    'create',
    context_settings={"ignore_unknown_options": True},
)
@click.argument("name")
@click.argument("source")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def create_profile(
    name: str,
    source: str,
    args: List[str] = None,
) -> None:
    """Creates a ZenNews profile.

    NAME denotes the name given to this profile.

    SOURCE is the name of the news source.

    ARGS are the additional parameters to configure the news source. (Each
    arg needs to be provided with either the '--key value' format or
    the '--key' format, in which case the value is True).

    For instance:

    NAME can be 'my_profile', SOURCE can be 'bbc' and the ARGS can include
    'news_tech' and 'sports_tennis'. In this case, the command needs to be
    structured as follows:

    'zennews create profile my_profile bbc --news_tech --sports_tennis'

    """
    # TODO: Make sure that a profile with the same name doesn't exist already
    # Parse the arguments
    parsed_args = parse_args(source=source, args=args)

    profile_obj = Profile(
        name=name,
        source=source,
        args=parsed_args,
    )

    save_profile(profile_obj)


@profile.command('remove')
@click.argument("name")
def remove_profile(name) -> None:
    """Removes a ZenNews profile."""
    # TODO: Load schedules and see if there is a schedule with the following
    #   profile
    delete_profile(name=name)


@profile.command('list')
def list_profiles() -> None:
    """Lists all created ZenNews profiles."""
    profiles = load_profiles()
    display_profiles(profiles)
