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

from typing import Dict


def score_risk(evaluation: Dict) -> Dict[str, float]:
    """Return dict with per‑factor risk ∈ [0,1] + overall."""
    # Example heuristics (tune for real use‑case)
    auc = evaluation["metrics"]["auc"]
    bias_flag = evaluation.get("bias_flag", False)
    disparity = max(
        abs(v.get("selection_rate_disparity", 0.0)) for v in evaluation["fairness"].values()
    )

    risk_auc = 1 - auc  # low AUC → higher risk
    risk_bias = 0.8 if bias_flag else disparity  # flat score if flagged
    overall = round(min(1.0, 0.5 * risk_auc + 0.5 * risk_bias), 3)

    return {
        "risk_auc": round(risk_auc, 3),
        "risk_bias": round(risk_bias, 3),
        "overall": overall,
    }
