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

# ruff: noqa: I001

from .logger import logger
from .prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    clean_text,
    format_prompt_for_deepseek,
)
from .json_parser import try_extract_json_from_text
from .classification_helpers import (
    prepare_classification_json,
    process_articles_parallel,
    process_articles_sequential,
)
from .classification_metrics import (
    calculate_and_save_metrics_from_json,
    calculate_metrics,
    create_metrics_report,
    log_metrics_to_console,
)
from .claude_evaluator import ClaudeEvaluator
from .remote_setup import determine_device, determine_if_remote
from .setup_environment import get_hf_token, with_setup_environment
from .training_eval_metrics import (
    calculate_memory_usage,
    calculate_prediction_costs,
    compute_classification_metrics,
    estimate_inference_cost,
    flatten_metrics,
    measure_inference_latency,
)
from .model_comparison_metrics import (
    evaluate_modernbert,
    calculate_claude_costs,
    prepare_metrics,
)
from .config_loaders import load_config
from .merge import get_identifier, transform_classification_results
from .docker_settings import apply_docker_settings
