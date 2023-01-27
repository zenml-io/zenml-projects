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

from constants import SUPPORTED_SOURCES
from models import Profile


def save_profile(profile: Profile) -> None:
    """Store the given profile."""


def load_profile(name: str) -> Profile:
    """Fetches a single profile by name."""


def load_profiles() -> List[Profile]:
    """Fetches all the defined profiles."""


def save_schedule() -> None:
    """Saves a schedule of an activated profile."""


def load_schedule(name: str) -> Schedule:
    """Fetches a schedule by name."""


def display_profiles(profiles: List[Profile]) -> None:
    """Displays the defined profiles."""


def parse_args(source: str, args: List[str]):
    """Auxiliary function to parse the additional arguments.

    Some examples:

    args: ["--a", "3", "--b", "4"] -> result: {"a": "3", "b": "4"}
    args  ["--a", "--b", "4"] -> result: {"a": True, "b": "4"}
    args  ["3", "--b", "4"] -> result: ValueError

    Arguments:
        source: str that denotes the news source
        args: list of additional arguments

    Raises:
        ValueError: if the additional arguments have an unsupported format.

    Returns:
        the "Parameters" for the selected new source
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

    # TODO: Investigate whether we should create the params here
    return SUPPORTED_SOURCES.get(source)(**results)
