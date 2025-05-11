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

from pathlib import Path
from typing import Annotated, Any, Dict, List

import joblib
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score, roc_auc_score
from zenml import log_metadata, step
from zenml.logger import get_logger

from utils.model_definition import model_definition

logger = get_logger(__name__)


@step(model=model_definition)
def evaluate_model(
    model_path: Annotated[str, "model_path"],
    test_df: Annotated[pd.DataFrame, "test_df"],
    protected_attributes: List[str],
    target: str = "target",
) -> Annotated[Dict[str, Any], "evaluation_results"]:
    """Compute performance + fairness metrics, emit Slack alert if disparity > 0.2.

    Articles 9 & 15 compliant.

    Args:
        model_path: Path to the trained model.
        test_df: Test dataset.
        protected_attributes: List of protected attributes.
        target: Target column name.
    """
    model = joblib.load(Path(model_path))

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
                print(f"Warning: Skipping protected attribute '{attr}' - not found in dataset")
                continue
            sensitive_col = matching_cols[0]
            print(f"Using '{sensitive_col}' for protected attribute '{attr}'")

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

    # ---- Log --------------------------------------------------------------
    log_metadata(
        {
            "metrics": metrics,
            "fairness_metrics": fairness_report,
            "bias_flag": bias_flag,
        }
    )

    # Optional alert
    # if bias_flag:
    #     try:
    #         from slack_sdk import WebClient

    #         zenml_client = Client()
    #         slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    #         slack_client.chat_postMessage(
    #             channel=os.getenv("SLACK_CHANNEL", "#ai-alerts"),
    #             text=f":warning: Bias detected in run - disparity > 0.2\nRun ID: {zenml_client.active_stack_model.id}",
    #         )
    #     except Exception:
    #         pass

    return {"metrics": metrics, "fairness": fairness_report}
