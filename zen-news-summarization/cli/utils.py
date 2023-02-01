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
import os.path
from typing import List, Dict, Tuple, TYPE_CHECKING, Optional, Any

import click
from rich import box, table, console

from cli.constants import PROFILES_PATH, CONFIG_PATH
from models import Profile, Config, Article

from zenml.client import Client

if TYPE_CHECKING:
    from zenml.steps import BaseStep, BaseParameters


def save_profile(profile: Profile) -> None:
    """Store the given profile.

    Args:
        profile: the profile object to store.

    Raises:
        ValueError, if a profile with the same name already exists.
    """
    filename = os.path.join(PROFILES_PATH, f'{profile.name}.json')

    if os.path.exists(filename):
        raise ValueError(
            f"A profile with the name '{profile.name}' already exists."
        )

    with open(filename, 'w') as f:
        json.dump(profile.json(), f)


def load_profile(name: str) -> Profile:
    """Fetches a single profile.

    Args:
        name: the name of the profile to load.

    Raises:
        ValueError, if there are no profiles with the given name.
    """
    filepath = os.path.join(PROFILES_PATH, f'{name}.json')

    if not os.path.exists(filepath):
        raise ValueError(f"No profiles exists with the name {name}")

    with open(filepath, 'r') as f:
        return Profile.parse_raw(json.load(f))


def load_profiles() -> List[Profile]:
    """Fetches all the defined profiles.

    Returns:
        a list of all defined profiles.
    """

    profiles = []

    for filename in os.listdir(PROFILES_PATH):
        filepath = os.path.join(PROFILES_PATH, filename)
        with open(filepath, 'r') as f:
            profiles.append(Profile.parse_raw(json.load(f)))

    return profiles


def delete_profile(name: str) -> None:
    """Deletes the profile with the given name.

    Args:
        name: the name of the profile to delete.

    Raises:
        ValueError, if there are no profiles with the given name.
    """
    filename = os.path.join(PROFILES_PATH, f'{name}.json')

    if os.path.exists(filename):
        os.remove(filename)
    else:
        raise ValueError(f"There are no profile with the name '{name}'.")


def display_profiles(profiles: List[Profile]) -> None:
    """Displays the defined profiles in a table.

    Args:
        profiles: the list of profiles to display in the table.
    """
    config = load_config()

    rich_table = table.Table(
        box=box.HEAVY_EDGE,
        title="ZenNews Profiles",
        caption="List of defined ZenNews profiles.",
        show_lines=True,
    )
    rich_table.add_column("ACTIVE", overflow="fold")
    rich_table.add_column("NAME", overflow="fold")
    rich_table.add_column("SOURCE", overflow="fold")
    rich_table.add_column("STACK", overflow="fold")
    rich_table.add_column("FREQUENCY", overflow="fold")
    rich_table.add_column("ARGS", overflow="fold")

    for profile in profiles:
        rich_table.add_row(
            "*" if profile.name in config.active_profiles else "",
            profile.name,
            profile.source,
            profile.stack,
            profile.frequency,
            '\n'.join([f"{k} = {v}" for k, v in profile.args.items()]),
        )

    console.Console(markup=True).print(rich_table)


def parse_args(args: List[str]) -> Dict[str, str]:
    """Auxiliary function to parse the additional arguments.

    Some examples:

    args: ["--a", "3", "--b", "4"] -> result: {"a": "3", "b": "4"}
    args  ["--a", "--b", "4"] -> result: {"a": True, "b": "4"}
    args  ["3", "--b", "4"] -> result: ValueError

    Arguments:
        args: list of additional arguments

    Raises:
        ValueError: if the additional arguments have an unsupported format.

    Returns:
        the dict of parsed arguments
    """
    results = {}

    last = None
    for current in args:
        if last is None:
            if current.startswith("--"):
                last = current
                continue
            else:
                raise ValueError(
                    f'The additional arguments have an unsupported format.'
                    f'Please make sure that each "value" is preceded by a key '
                    f'defined as "--key".'
                )

        if last is not None:
            if current.startswith("--"):
                results[last] = True
                last = current
            else:
                results[last] = current
                last = None

    if last is not None:
        results[last] = True

    return results


def load_config() -> Config:
    """Loads the ZenNews config.

    Returns:
        the global config object.
    """
    with open(CONFIG_PATH, 'r') as f:
        return Config.parse_raw(json.load(f))


def save_config(config: Config) -> None:
    """Saves the ZenNews config.

    Args:
        config: the config object to store.
    """
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config.json(), f)


def load_step_and_parameters(profile) -> Tuple["BaseStep", "BaseParameters"]:
    """Fetches the source step and the related parameters class of a profile.

    Args:
        profile: the profile object.
    """

    from steps import SOURCE_STEP_MAPPING

    step_dict = SOURCE_STEP_MAPPING[profile.source]

    return step_dict['step'], step_dict['parameters']


def error(text: str) -> None:
    """Wrapper around the click.ClickException.

    Args:
        text, str, the exception text.

    Raises:
        click.ClickException with the defined style.
    """
    raise click.ClickException(message=click.style(text, fg="red", bold=True))


def warning(text: str) -> None:
    """Wrapper around the 'click.echo' for warning messages.

    Args:
        text, str, the warning message.
    """
    click.secho(text, fg='yellow', bold=True)


class stack_handler(object):
    """Context manager that switches the active stack temporarily."""

    def __init__(self, target_stack_name: str = 'default') -> None:
        """Initialization of the stack handler.

        Args:
            target_stack_name: str, the name of the target stack
        """
        self.active_stack_name = None
        self.target_stack_name = target_stack_name

    def __enter__(self) -> "stack_handler":
        """Enter function of the stack handler.

        Saves the name of the current active stack and activates the temporary
        target stack.

        Returns:
            the handler instance.
        """
        client = Client()

        self.active_stack_name = client.active_stack_model.name
        client.activate_stack(self.target_stack_name)
        return self

    def __exit__(
        self,
        type_: Optional[Any],
        value: Optional[Any],
        traceback: Optional[Any],
    ) -> Any:
        """Exit function of the stack handler.

        Sets the previous stack as the active stack again.

        Args:
            type_: The class of the exception
            value: The instance of the exception
            traceback: The traceback of the exception
        """

        client = Client()
        client.activate_stack(self.active_stack_name)


def display_articles(articles: List[Article]) -> None:
    """Display the articles on the CLI."""
    # TODO: Implement a pretty print for the articles
    click.echo(articles)
