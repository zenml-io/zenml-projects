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
from typing import List, Dict

from rich import box, table, console

from cli.constants import PROFILES_PATH
from models import Profile


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
        raise ValueError("The profile does not exist.")


def display_profiles(profiles: List[Profile]) -> None:
    """Displays the defined profiles in a table.

    Args:
        profiles: the list of profiles to display in the table.
    """

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
            "",  # TODO: Add active status
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
