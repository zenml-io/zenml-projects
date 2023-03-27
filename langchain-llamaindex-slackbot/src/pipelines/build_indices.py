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

import datetime
import os
import shutil
from typing import List, Optional, Tuple
from uuid import uuid4

import git
import requests
from tqdm import tqdm
from zenml.pipelines import pipeline

from steps.gitbook_docs_loader import DocsLoaderParameters, docs_loader
from steps.index_generator import index_generator
from steps.slack_loader import (
    SLACK_CHANNEL_IDS,
    SlackLoaderParameters,
    slack_loader,
)


def get_zenml_versions():
    # Clone zenml repo to random dir in CWD
    random_repo_name = f"_{uuid4()}"
    repo_dir = os.path.join(os.getcwd(), random_repo_name)
    repo = git.Repo.clone_from(
        url="https://github.com/zenml-io/zenml",
        to_path=random_repo_name,
    )

    # Get all release versions
    versions = []
    for refs in repo.remote().refs:
        branch_name = refs.name
        if not branch_name.startswith("origin/release"):
            continue
        version = branch_name.split("/")[-1]
        versions.append(version)

    # Cleanup
    shutil.rmtree(repo_dir)

    return versions


def _page_exists(url: str) -> bool:
    import requests

    r = requests.get(url)
    return r.status_code == 200


def get_release_date(
    package_name: str, version: str
) -> Tuple[datetime.datetime, Optional[datetime.datetime]]:
    """Get the release date of a package version.

    Args:
        package_name: Name of the package.
        version: Version of the package.

    Returns:
        The release date of the package version, and the date of the next
        release if it exists (or None).
    """
    # Get the package's release information from the PyPI API
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")

    # Parse the JSON data
    data = response.json()

    # Get a list of the package's release versions
    release_info = data["releases"].get(version)

    if not release_info:
        raise ValueError(
            f"Version {version} not found for package {package_name}."
        )
    release_upload_time = datetime.datetime.strptime(
        data["releases"][version][0]["upload_time"], "%Y-%m-%dT%H:%M:%S"
    )

    two_weeks_later = release_upload_time + datetime.timedelta(weeks=2)
    if two_weeks_later > datetime.datetime.now():
        two_weeks_later = datetime.datetime.now()

    return (
        release_upload_time,
        two_weeks_later,
    )


def build_indices_for_zenml_versions(
    versions: List[str], pipeline_name="zenml_docs_index_generation"
):
    @pipeline(name=pipeline_name)
    def docs_to_index_pipeline(document_loader, slack_loader, index_generator):
        documents = document_loader()
        slack_docs = slack_loader()
        index_generator(documents, slack_docs)

    for version in tqdm(versions):
        base_url = "https://docs.zenml.io"
        docs_url = f"https://docs.zenml.io/v/{version}"
        if not _page_exists(docs_url):
            print(f"Couldn't find docs page for zenml version '{version}'.")
            continue
        print(f"Building index for zenml docs of version '{version}'...")
        release_date, next_release_date = get_release_date("zenml", version)

        pip = docs_to_index_pipeline(
            document_loader=docs_loader(
                params=DocsLoaderParameters(
                    docs_uri=docs_url, base_url=base_url
                )
            ),
            index_generator=index_generator(),
            slack_loader=slack_loader(
                params=SlackLoaderParameters(
                    channel_ids=SLACK_CHANNEL_IDS,
                    earliest_date=release_date,
                    latest_date=next_release_date,
                )
            ),
        )
        run_name = (
            f"{pipeline_name}" + "_{{{date}}}_{{{time}}}" + f"_{version}"
        )
        try:
            pip.run(run_name=run_name)
        except Exception as e:
            print(f"Failed to build index for zenml version '{version}'.")
            print(e)
