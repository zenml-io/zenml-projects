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
"""Module for extracting JSON from a response, handling various formats."""

import json
import re
from typing import Any, Dict


def _ensure_string_raw_text(result: Dict) -> Dict:
    """Ensure raw_text in result is a string and not a list.

    Args:
        result: Dictionary that may contain a raw_text field

    Returns:
        Dictionary with raw_text converted to string if needed
    """
    if "raw_text" in result and isinstance(result["raw_text"], list):
        result["raw_text"] = "\n".join(result["raw_text"])
    return result


def try_extract_json_from_response(response: Any) -> Dict:
    """Extract JSON from a response, handling various formats efficiently.

    Args:
        response: The response which could be a string, dict, or an object with content.

    Returns:
        Dict with extracted data.
    """
    if isinstance(response, dict) and "raw_text" in response:
        return _ensure_string_raw_text(response)

    response_text = ""
    if hasattr(response, "choices") and response.choices:
        msg = getattr(response.choices[0], "message", None)
        if msg and hasattr(msg, "content"):
            response_text = msg.content
    elif isinstance(response, str):
        response_text = response
    elif hasattr(response, "raw_text"):
        raw_text = response.raw_text
        result = {
            "raw_text": raw_text,
            "confidence": getattr(response, "confidence", None),
        }
        return _ensure_string_raw_text(result)

    response_text = response_text.strip()

    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            return _ensure_string_raw_text(parsed)
    except json.JSONDecodeError:
        pass

    json_block = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_block:
        json_str = json_block.group(1).strip()
        try:
            parsed = json.loads(json_str)
            return _ensure_string_raw_text(parsed)
        except json.JSONDecodeError:
            pass

    json_substring = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_substring:
        json_str = json_substring.group(0).strip()
        try:
            parsed = json.loads(json_str)
            return _ensure_string_raw_text(parsed)
        except json.JSONDecodeError:
            pass

    return {"raw_text": response_text, "confidence": None}
