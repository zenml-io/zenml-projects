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
from typing import Any, Dict

from zenml import step


@step
def save_comparison_metrics(metrics: Dict[str, Any], config: Dict[str, Any]):
    """
    Save comparison metrics as a structured JSON file.

    Args:
        metrics: Dictionary of metrics
        config: Dictionary of configuration
    """
    metrics_dir = Path("model_compare_metrics")
    metrics_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(metrics_dir / f"comparison_metrics_{timestamp}.json", "w") as f:
        json.dump(
            {
                "config": config,
                "metrics": metrics,
                "timestamp": timestamp,
            },
            f,
            indent=2,
        )
