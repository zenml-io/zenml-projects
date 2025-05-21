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

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jinja2
import yaml


def setup_jinja_environment(template_dir: Path) -> jinja2.Environment:
    """Set up Jinja2 environment with custom filters for Annex IV template rendering.

    Args:
        template_dir: Directory containing Jinja2 templates

    Returns:
        Configured Jinja2 environment
    """
    loader = jinja2.FileSystemLoader(searchpath=str(template_dir))
    env = jinja2.Environment(loader=loader)

    # Add custom filters
    env.filters["join_kv"] = _filter_join_kv
    env.filters["join_list_kv"] = _filter_join_list_kv
    env.filters["list_keys"] = _filter_list_keys
    env.filters["format_inputs"] = _filter_format_inputs
    env.filters["format_outputs"] = _filter_format_outputs
    env.filters["safe_str"] = _filter_safe_str
    env.filters["to_yaml"] = _filter_to_yaml
    env.filters["nl2br"] = _filter_nl2br
    env.filters["preserve_newlines"] = _filter_preserve_newlines

    return env


def load_sample_inputs(template_dir: Path) -> Dict[str, Any]:
    """Load sample inputs from the sample_inputs.json file.

    Args:
        template_dir: Directory containing the template and sample inputs file

    Returns:
        Dictionary with sample input data
    """
    sample_inputs_path: Path = template_dir / "sample_inputs.json"
    if sample_inputs_path.exists():
        with open(sample_inputs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def render_annex_iv_template(
    metadata: Dict[str, Any],
    manual_inputs: Optional[Dict[str, Any]] = None,
    template_dir: Optional[Path] = None,
) -> str:
    """Render the Annex IV Jinja template with collected metadata.

    Args:
        metadata: ZenML metadata dictionary
        manual_inputs: Manual inputs dictionary (optional)
        template_dir: Directory containing the template (optional)

    Returns:
        Rendered template content
    """
    if template_dir is None:
        template_dir = Path(__file__).parent.parent.parent / "docs" / "templates"

    env: jinja2.Environment = setup_jinja_environment(template_dir)
    template: jinja2.Template = env.get_template("annex_iv_template.j2")

    # Load sample inputs if manual_inputs is not provided
    if manual_inputs is None:
        manual_inputs = load_sample_inputs(template_dir)

    # Set up the template variables
    template_data: Dict[str, Any] = {
        **metadata,
        "manual_inputs": manual_inputs,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return template.render(**template_data)


# Custom filter functions
def _filter_join_kv(d: Dict[str, Any]) -> str:
    """Format dictionary key-value pairs to a readable string."""
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else "-"


def _filter_join_list_kv(d: Dict[str, List[Any]]) -> str:
    """Format dictionaries with list values (ensuring all elements are strings)."""
    return ", ".join(f"{k}=[{','.join(str(x) for x in v)}]" for k, v in d.items()) if d else "-"


def _filter_list_keys(d: Dict[str, Any]) -> str:
    """Extract dictionary keys."""
    return ", ".join(d.keys()) if isinstance(d, dict) and d else "-"


def _filter_format_inputs(inputs: Dict[str, Any]) -> str:
    """Format inputs for pipeline steps with better error handling."""
    if not inputs:
        return "-"

    result: List[str] = []
    for k, v in inputs.items():
        try:
            # Only take the first 8 characters of the string representation
            value_str: str = str(v)[:8]
            result.append(f"{k}=`{value_str}`")
        except Exception:
            # Fallback in case of errors
            result.append(f"{k}=...")

    return ", ".join(result)


def _filter_format_outputs(outputs: Dict[str, Union[Any, List[Any]]]) -> str:
    """Format outputs for pipeline steps with better handling of list values."""
    if not outputs:
        return "-"

    result: List[str] = []
    for k, v in outputs.items():
        if isinstance(v, list) and v:
            # Handle list of IDs
            formatted_ids: str = ",".join(f"`{str(x)[:8]}`" for x in v)
            result.append(f"{k}=[{formatted_ids}]")
        else:
            # Handle single value or empty list
            result.append(f"{k}=`{str(v)[:8]}`" if v else f"{k}=-")

    return ", ".join(result)


def _filter_safe_str(obj: Any) -> str:
    """Safe string converter for any type."""
    return str(obj) if obj is not None else "-"


def _filter_to_yaml(obj: Any) -> str:
    """Dump to yaml with nice formatting."""
    return yaml.dump(obj, default_flow_style=False)


def _filter_nl2br(text: Optional[str]) -> str:
    """Handle newlines in Jinja templates by converting to HTML breaks."""
    return text.replace("\n", "<br>") if text else ""


def _filter_preserve_newlines(text: Optional[str]) -> str:
    """Preserve newlines for markdown."""
    return text if text else ""
