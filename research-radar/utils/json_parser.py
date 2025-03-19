# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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
#

import contextlib
import json
import re

JSON_PATTERN = re.compile(r"```json\n(.*?)```", re.DOTALL)
DIRECT_JSON_PATTERN = re.compile(r"\{[^}]*\}", re.DOTALL)
CLEANED_JSON_PATTERN = re.compile(r'("[^"]*":\s*)"([^"]*)"')


def try_extract_json_from_text(text: str) -> tuple[str, dict | None]:
    """
    Try to extract a JSON object from a string.

    Args:
        text: The text to extract the JSON object from

    Returns:
        Tuple of the original text and the JSON object.
        If no JSON object is found, returns a tuple of the original text and None.
    """
    if match := JSON_PATTERN.search(text):
        json_results = match.group(1)
        with contextlib.suppress(json.JSONDecodeError):
            return text, json.loads(json_results)
    if match := DIRECT_JSON_PATTERN.search(text):
        json_text = match.group(0)
        with contextlib.suppress(json.JSONDecodeError):
            return text, json.loads(json_text)
    return text, None
