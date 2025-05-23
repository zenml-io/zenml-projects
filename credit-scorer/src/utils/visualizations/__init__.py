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

"""HTML component utilities for rendering compliance dashboards."""

from .eval import generate_eval_visualization
from .whylogs import generate_whylogs_visualization
from .dashboard import generate_compliance_dashboard_html

__all__ = [
    "generate_eval_visualization",
    "generate_whylogs_visualization",
    "generate_compliance_dashboard_html",
]