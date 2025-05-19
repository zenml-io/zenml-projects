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
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

from src.constants import (
    EVALUATION_RESULTS_NAME,
    MODEL_NAME,
    RELEASES_DIR,
    TARGET_COLUMN,
    TEST_DATASET_NAME,
)
from src.utils import (
    analyze_fairness,
    report_bias_incident,
    save_artifact_to_modal,
)
from src.utils.model_definition import model_definition

logger = get_logger(__name__)


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle UUIDs."""

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


@step(model=model_definition)
def evaluate_model(
    protected_attributes: List[str],
    test_df: Annotated[pd.DataFrame, TEST_DATASET_NAME],
    target: str = TARGET_COLUMN,
    model: Annotated[Any, MODEL_NAME] = None,
) -> Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME]:
    """Compute performance + fairness metrics. Articles 9 & 15 compliant.

    Args:
        protected_attributes: List of protected attributes
        test_df: Test dataset
        target: Target column name
        model: The trained model

    Returns:
        Dictionary containing evaluation and fairness results
    """
    # Get model and identify target column
    if model is None:
        client = Client()
        model = client.get_artifact_version(name_id_or_prefix=MODEL_NAME)

    target_col = next(
        col for col in test_df.columns if col.endswith(f"__{target}") or col == target
    )
    # Prepare test data
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # Evaluate model performance
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    performance_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    # Analyze fairness across protected attributes
    fairness_metrics, bias_flag = analyze_fairness(
        y_test,
        y_pred,
        protected_attributes,
        test_df,
    )

    # Create fairness report
    fairness_report = {
        "metrics": performance_metrics,
        "fairness_metrics": fairness_metrics,
        "bias_flag": bias_flag,
        "protected_attributes_checked": protected_attributes,
    }

    # Get the run ID for artifacts
    run_id = str(get_step_context().pipeline_run.id)

    # Save report to Modal
    release_dir = f"{RELEASES_DIR}/{run_id}"
    Path(release_dir).mkdir(parents=True, exist_ok=True)
    fairness_file_path = f"{release_dir}/fairness_report.json"

    save_artifact_to_modal(
        artifact=fairness_report,
        artifact_path=fairness_file_path,
    )

    logger.info(f"Fairness report saved to Modal Volume: {fairness_file_path}")

    # Log metadata for the pipeline
    log_metadata(
        {
            "metrics": performance_metrics,
            "bias_flag": bias_flag,
            "fairness_file_path": fairness_file_path,
        }
    )

    # Create incident report if bias detected
    if bias_flag:
        report_bias_incident(fairness_report, run_id)

    # Return the evaluation results
    return {"metrics": performance_metrics, "fairness": fairness_report}
