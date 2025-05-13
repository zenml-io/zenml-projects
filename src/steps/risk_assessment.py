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

import tempfile
from pathlib import Path
from typing import Annotated, Dict

import pandas as pd
from openpyxl import Workbook, load_workbook
from zenml import get_step_context, log_metadata, step

from src.constants import (
    MODAL_COMPLIANCE_DIR,
    MODAL_MANUAL_FILLS_DIR,
    MODAL_RISK_REGISTER_PATH,
    RISK_SCORES_NAME,
)
from src.utils import score_risk
from src.utils.modal_utils import save_artifact_to_modal

# --------------------------------------------------------------------------- #
RiskScores = Annotated[Dict[str, float], RISK_SCORES_NAME]
# --------------------------------------------------------------------------- #


@step
def risk_assessment(evaluation_results: Dict) -> RiskScores:
    """Compute risk scores & update register. Article 9 compliant.

    Converts evaluation metrics + bias flag → quantitative risk score
    Updates risk_register.xlsx (one row / model version)
    Logs summary via `log_metadata`

    Args:
        evaluation_results: Dictionary containing evaluation results.

    Returns:
        Dictionary containing risk scores.
    """
    scores = score_risk(evaluation_results)

    # Build or update the workbook in memory
    wb_path = Path(tempfile.mkdtemp()) / "risk_register.xlsx"
    if not wb_path.exists():
        wb = Workbook()
        ws = wb.active
        ws.title = "Risks"
        ws.append(["Run_ID", "Risk_overall", "Risk_auc", "Risk_bias", "Status"])
    else:
        wb = load_workbook(wb_path)
        ws = wb["Risks"]

    # Get run_id from step context
    run_id = get_step_context().pipeline_run.id

    # Check if this run already logged
    run_ids = [cell.value for cell in ws["A"][1:]]  # skip header
    if run_id in run_ids:
        row_idx = run_ids.index(run_id) + 2
    else:
        row_idx = ws.max_row + 1

    ws.cell(row=row_idx, column=1).value = str(run_id)
    ws.cell(row=row_idx, column=2).value = scores["overall"]
    ws.cell(row=row_idx, column=3).value = scores["risk_auc"]
    ws.cell(row=row_idx, column=4).value = scores["risk_bias"]
    ws.cell(row=row_idx, column=5).value = (
        "Mitigation needed" if scores["overall"] > 0.4 else "Acceptable"
    )

    wb.save(wb_path)  # write into /tmp/risk_register.xlsx

    # Save risk register to Modal Volume
    save_artifact_to_modal(
        artifact=wb,
        artifact_path=MODAL_RISK_REGISTER_PATH,
    )

    # 3) (Optional) snapshot as Markdown and upload
    df = pd.read_excel(wb_path, sheet_name="Risks")
    md = df.to_markdown(index=False)
    save_artifact_to_modal(
        artifact=md,
        artifact_path=f"{MODAL_MANUAL_FILLS_DIR}/risk_register.md",
        overwrite=True,
    )

    # Log metadata
    log_metadata({"risk_scores": scores})

    return scores
