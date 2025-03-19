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

import os
from typing import Dict

from schemas import zenml_project
from steps import compare_models, load_test_set, save_comparison_metrics
from utils import load_config
from zenml import pipeline


@pipeline(model=zenml_project, enable_cache=False)
def model_comparison_pipeline(config: Dict | None = None):
    """Compare ModernBERT and Claude Haiku performance.

    Args:
        config: Pipeline configuration containing:
            - Model paths and batch sizes
            - Cost parameters
            - Evaluation settings
    """
    config = config or load_config()

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError(
            "Anthropic API key not found in environment (ANTHROPIC_API_KEY)"
        )

    dataset_config = config["steps"]["compare"]["dataset"]
    test_dataset = load_test_set(
        source_type=dataset_config["source_type"],
        path=dataset_config["path"],
        artifact_name=dataset_config["artifact_name"],
        version=dataset_config["version"],
    )

    metrics = compare_models(
        test_dataset=test_dataset,
        anthropic_api_key=anthropic_api_key,
        modernbert_path=config["outputs"]["ft_model"],
        tokenizer_path=config["outputs"]["ft_tokenizer"],
        modernbert_batch_size=config["steps"]["compare"]["batch_sizes"][
            "modernbert"
        ],
        claude_batch_size=config["steps"]["compare"]["batch_sizes"]["claude"],
        claude_haiku_token_costs=config["steps"]["compare"]["costs"][
            "claude_haiku"
        ],
    )

    run_config = {
        "artifact_name": "test_set",
        "batch_sizes": config["steps"]["compare"]["batch_sizes"],
        "random_seed": 42,
    }

    save_comparison_metrics(metrics=metrics, config=run_config)
