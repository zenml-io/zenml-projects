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

from pipelines.index_builder import docs_to_index_pipeline


def main():
    docs_url = "https://docs.zenml.io"
    website_url = "https://zenml.io"
    repo_url = "https://github.com/zenml-io/zenml/tree/main/examples"
    release_notes_url = (
        "https://github.com/zenml-io/zenml/blob/main/RELEASE_NOTES.md"
    )

    docs_to_index_pipeline(
        website_url=website_url,
        docs_url=docs_url,
        repo_url=repo_url,
        release_notes_url=release_notes_url,
    )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    logging.getLogger().setLevel(logging.INFO)
    main()
