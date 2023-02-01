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
from zenml.enums import StackComponentType

from cli.base import cli
from cli.constants import SUPPORTED_SOURCES, SUPPORTED_ORCHESTRATORS
from cli.utils import (
    parse_args,
    save_profile,
    load_profiles,
    delete_profile,
    display_profiles,
    load_config,
    save_config,
    load_profile,
warning,
error,
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

    # TODO: Handle the validation errors here
    source_args = SUPPORTED_SOURCES.get(source)(**parsed_args)

    try:
        save_profile(
            Profile(
                name=name,
                source=source,
                args=source_args,
                stack=stack,
                frequency=frequency,
            )
        )
    except ValueError as e:
        raise click.ClickException(message=click.style(e, fg="red", bold=True))

    click.secho(f"Successfully created profile: '{name}'!", fg='green')


@profile.command('remove')
@click.argument("name")
def remove_profile(name) -> None:
    """Removes a ZenNews profile."""
    # TODO: Check whether the profile is active first
    try:
        delete_profile(name=name)
    except ValueError as e:
        raise click.ClickException(message=click.style(e, fg="red", bold=True))

    click.secho(f"Successfully removed profile: '{name}'!", fg='green')


@profile.command('list')
def list_profiles() -> None:
    """Lists all created ZenNews profiles."""
    profiles = load_profiles()
    display_profiles(profiles)


@profile.command("update")
@click.argument("name")
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
    source,
    stack,
    frequency,
    args,
) -> None:
    """Updates a ZenNews profile."""

    config = load_config()
    if name in config.active_profiles:
        raise click.ClickException(
            message=click.style(
                text="Active profiles can not be updated. Please deactivate "
                     "your profile before making any changes.",
                fg="red",
                bold=True,
            )
        )
    profile_obj = load_profile(name)

    if source:
        profile_obj.source = source

    if stack:
        profile_obj.stack = stack

    if frequency:
        profile_obj.frequency = frequency

    if args:
        if profile_obj.args:
            profile_obj.args.update(args)
        else:
            profile_obj.args = args

    try:
        delete_profile(name=name)
        save_profile(profile=profile_obj)
    except ValueError as e:
        raise click.ClickException(message=click.style(e, fg="red", bold=True))

    click.secho(f"Successfully updated profile: '{name}'!", fg='green')


@profile.command("activate")
@click.argument("name")
def activate_profile(name) -> None:
    """Activates a ZenNews profile."""
    # TODO: Check if stack is present and validated
    # TODO: Check whether stack has an orchestrator with schedule support
    # TODO: Check whether stack has an alerter
    # TODO: Check whether frequency is correctly formatted.
    config = load_config()

    profile = load_profile(name)

    from zenml.client import Client
    client = Client()

    stack = client.get_stack(name)
    components = stack.components

    orchestrator = components[StackComponentType.ORCHESTRATOR][0]

    if orchestrator.flavor not in SUPPORTED_ORCHESTRATORS:
        error('Does not work.')

    if StackComponentType.ALERTER not in components:
        warning('There is no alerter!')

    config.active_profiles.add(name)
    save_config(config)
    click.secho(f"Successfully activated profile: '{name}'!", fg='green')
    from zenml.config.schedule import Schedule




