# SOP – Model Release & Deployment Approval

_Version 0.1 • Owner: **AI Compliance Officer**_

| Section     | Detail                                                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Purpose** | Ensure each new model version satisfies technical KPIs and EU-AI-Act obligations before it is deployed.                    |
| **Scope**   | All models produced by `train_model` step in the credit-scoring workflow.                                                  |
| **Roles**   | **Submitter:** ML Engineer • **Tester:** QA Lead • **Approver:** AI Compliance Officer                                     |
| **Inputs**  | Pipeline **run-ID**, evaluation & fairness metrics, risk scores, Annex IV PDF                                              |
| **Records** | Approval JSON in Modal Volume `compliance/approvals/` • Entry in `model_release_log.md` • `deployment_url` in run metadata |

---

## 1 Release Flow

| Step | Action                                                                                                                                                       | Responsible        | Evidence           |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ | ------------------ |
| 1    | Trigger `training_pipeline`; open Jira ticket **AI-REL-<run-id>**                                                                                            | ML Eng.            | Pipeline run page  |
| 2    | `approve_deployment` step displays metrics & risk, then waits for decision                                                                                   | CI job             | Terminal / Slack   |
| 3    | **Tester review** – QA Lead checks Annex IV & metrics, adds “Tester ✓” comment                                                                               | QA Lead            | Jira comment       |
| 4    | **Approval** – Compliance Officer:<br>• Interactive: type `y` + rationale<br>• Automated: set env vars `DEPLOY_APPROVAL=y`, `APPROVER`, `APPROVAL_RATIONALE` | Compliance Officer | JSON file (see §3) |
| 5    | `deploy_model` uploads model and writes `deployment_url` to run metadata                                                                                     | Pipeline step      | Metadata key       |
| 6    | ML Eng. appends row to `model_release_log.md` (model path, checksum, URL)                                                                                    | ML Eng.            | Git commit         |

---

## 2 Pass / Fail Criteria

(_Configured in `src/constants.py` under "Model Approval Thresholds" – adjust to business needs._)

| Category           | Threshold (default) |
| ------------------ | ------------------- |
| Accuracy           | ≥ **0.80**          |
| Bias disparity     | ≤ **0.20**          |
| Overall risk score | ≤ **0.40**          |
| Fairness flags     | None                |

If any threshold fails and the approver selects **Reject**, the step raises
`RuntimeError` and the pipeline stops.

---

## 3 Approval Record Structure

Saved automatically by `approve_deployment` to Modal Volume `compliance/approvals/approval_<timestamp>.json`.

```jsonc
{
  "approval_id": "approval_2025-05-14T10-03-55",
  "timestamp": "2025-05-14T10:03:55Z",
  "decision_mode": "interactive | automated",
  "approved": true,
  "approver": "Jane Doe",
  "rationale": "Meets KPIs; risk acceptable",
  "threshold_checks": {
    "Accuracy": true,
    "Bias disparity": true,
    "Risk score": true
  },
  "evaluation_summary": {
    "accuracy": 0.82,
    "auc": 0.91,
    "f1": 0.79,
    "fairness_flags": []
  },
  "risk_summary": {
    "risk_level": "low",
    "high_risk_factors": []
  },
  "deployment_url": "https://some-modal-url.modal.run"
}
```

## 4 Roll-back Procedure

1. Locate previous `deployment_url` in `model_release_log.md`.
2. Run `scripts/redeploy.py --url <old_url>` to restore the model.
3. Add “ROLLED BACK” label to the Jira ticket and note reason.

---

## 5 Review & Maintenance

| Activity                       | Frequency                  | Owner               |
| ------------------------------ | -------------------------- | ------------------- |
| Threshold review               | Quarterly                  | ML Engineering Lead |
| Spot-check approval JSON files | Quarterly                  | QA Lead             |
| SOP update                     | Annually or after incident | Compliance Officer  |

---

_Last updated: **[YYYY-MM-DD]** • Next review: **[YYYY-MM-DD]**_
