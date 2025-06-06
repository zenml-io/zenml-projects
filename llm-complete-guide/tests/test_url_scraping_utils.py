# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from steps.url_scraping_utils import extract_parent_section


@pytest.mark.parametrize(
    "url, expected_parent_section",
    [
        (
            "https://docs.zenml.io/user-guides/starter-guide/create-an-ml-pipeline",
            "user-guide",
        ),
        (
            "https://docs.zenml.io/v/docs/user-guides/production-guide/deploying-zenml",
            "user-guide",
        ),
        (
            "https://docs.zenml.io/stacks",
            "stacks-and-components",
        ),
    ],
)
def test_extract_parent_section(url, expected_parent_section):
    assert extract_parent_section(url) == expected_parent_section
