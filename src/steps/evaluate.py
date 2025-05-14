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

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List

import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score, roc_auc_score
from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

from src.constants import (
    EVALUATION_RESULTS_NAME,
    MODAL_FAIRNESS_DIR,
    MODEL_NAME,
    TARGET_COLUMN,
    TEST_DATASET_NAME,
)
from src.utils.modal_utils import save_artifact_to_modal
from src.utils.model_definition import model_definition

logger = get_logger(__name__)


@step(model=model_definition)
def evaluate_model(
    protected_attributes: List[str],
    test_df: Annotated[pd.DataFrame, TEST_DATASET_NAME],
    target: str = TARGET_COLUMN,
    model: Annotated[Any, MODEL_NAME] = None,
) -> Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME]:
    """Compute performance + fairness metrics, emit Slack alert if disparity > 0.2.

    Articles 9 & 15 compliant.

    Args:
        protected_attributes: List of protected attributes.
        test_df: Test dataset.
        target: Target column name.
        model: The trained model
    """
    # Use model if provided, otherwise fetch from ZenML
    if model is None:
        # Fetch the model from ZenML artifact store
        client = Client()
        model = client.get_artifact_version(name_id_or_prefix=MODEL_NAME)

    # data preprocessor set may have added a suffix to the target column
    target_col = next(
        col for col in test_df.columns if col.endswith(f"__{target}") or col == target
    )

    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    # ---- Fairness per protected attr --------------------------------------
    fairness_report = {}
    bias_flag = False

    for attr in protected_attributes:
        if attr in test_df.columns:
            sensitive_col = attr
        else:
            # Try to find a matching column
            matching_cols = [col for col in test_df.columns if attr in col]
            if not matching_cols:
                logger.warning(
                    f"Warning: Skipping protected attribute '{attr}' - not found in dataset"
                )
                continue
            sensitive_col = matching_cols[0]
            logger.info(f"Using '{sensitive_col}' for protected attribute '{attr}'")

        # use the matched column
        frame = MetricFrame(
            metrics={
                "selection_rate": selection_rate,
                "accuracy": accuracy_score,
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=test_df[sensitive_col],
        )
        disparity = frame.difference(method="between_groups")["selection_rate"]
        fairness_report[sensitive_col] = {
            "selection_rate_by_group": frame.by_group["selection_rate"].to_dict(),
            "accuracy_by_group": frame.by_group["accuracy"].to_dict(),
            "selection_rate_disparity": disparity,
        }

        if abs(disparity) > 0.2:  # configurable
            bias_flag = True

    # ---------- save full report to Modal Volume ----------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fairness_dir = Path(MODAL_FAIRNESS_DIR)
    fairness_file_path = fairness_dir / f"fairness_report_{timestamp}.json"
    fairness_report = {
        "metrics": metrics,
        "fairness_metrics": fairness_report,
        "bias_flag": bias_flag,
        "protected_attributes_checked": protected_attributes,
        "timestamp": timestamp,
    }

    save_artifact_to_modal(
        artifact=fairness_report,
        artifact_path=fairness_file_path,
    )

    logger.info(f"Fairness report saved to Modal Volume: {fairness_file_path}")

    # ---- Log --------------------------------------------------------------
    log_metadata(
        {
            "metrics": metrics,
            "bias_flag": bias_flag,
            "fairness_file_path": str(fairness_file_path),
        }
    )

    # Alert on bias detection
    if bias_flag:
        try:
            # Use the incident reporting API
            incident_data = {
                "severity": "high",
                "description": "Bias detected in model evaluation",
                "details": f"Disparity > 0.2 detected in protected attributes: {', '.join(protected_attributes)}",
                "source": "evaluate_model",
                "run_id": get_step_context().pipeline_run.id,
            }

            # Log from internal pipeline (doesn't go through API endpoint)
            from src.utils.incidents import create_incident_report

            create_incident_report(incident_data)

        except Exception as e:
            logger.warning(f"Failed to report bias incident: {e}")

    return {"metrics": metrics, "fairness": fairness_report}
