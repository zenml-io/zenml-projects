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

"""Utility functions for the project."""

from .compliance.annex_iv import (
    collect_zenml_metadata,
    generate_readme,
    load_and_process_manual_inputs,
    record_log_locations,
    write_git_information,
)
from .eval import analyze_fairness
from .preprocess import (
    DeriveAgeFeatures,
    DropIDColumn,
)
from .storage import (
    save_artifact_to_modal,
    save_evaluation_artifacts,
    save_visualizations,
)
from .visualizations import (
    generate_eval_visualization,
    generate_whylogs_visualization,
)
