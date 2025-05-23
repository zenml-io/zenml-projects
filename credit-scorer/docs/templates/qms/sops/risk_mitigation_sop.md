# SOP – Risk Assessment & Mitigation

_Version 0.1 • Owner: **AI Compliance Officer**_

| Section     | Detail                                                                                     |
| ----------- | ------------------------------------------------------------------------------------------ |
| **Purpose** | Fulfill EU-AI-Act Art 9: identify, score and mitigate risks before each deployment.        |
| **Scope**   | All credit-scoring models produced by `training_pipeline`.                                 |
| **Roles**   | ML Engineer (runs pipeline) · Data Gov. Manager (advisor) · Compliance Officer (approver). |
| **Inputs**  | `evaluation_results` dict from `evaluate_model` step.                                      |
| **Outputs** | Updated `risk_register.xlsx`, Jira “AI-Risk” ticket(s).                                    |

---

## 1 Automated Risk Scoring

Executed in `steps/risk_assessment.py`:

- Calculates **AUC risk** (`1 - auc`), **Bias risk** (max selection-rate disparity).
- Combines to `overall` score; writes row to `compliance/risk_register.xlsx`.
- Logs summary + hazard IDs in run metadata (`hazards` key).

Scoring thresholds (edit in `src/score_risk.py`):

| Metric    | Medium | High   |
| --------- | ------ | ------ |
| AUC risk  | > 0.15 | > 0.25 |
| Bias risk | > 0.10 | > 0.20 |
| Overall   | > 0.20 | > 0.40 |

---

## 2 Hazard Identification

Uses `HAZARD_DEFINITIONS` map in `src/hazards.py` to auto-flag:

- `bias_protected_groups` – disparity > 0.2
- `low_accuracy` – accuracy < 0.75
- … (extend as needed)

Each triggered hazard is stored in the `hazards` column of the risk register with a suggested mitigation.

---

## 3 Mitigation Actions

| Risk level    | Required action                            | Deadline              | Sign-off           |
| ------------- | ------------------------------------------ | --------------------- | ------------------ |
| **High** 🔴   | Implement mitigation _before deployment_   | Blocking              | Compliance Officer |
| **Medium** ⚠️ | Add mitigation plan to Jira within 30 days | 30 d                  | ML Lead            |
| **Low** ℹ️    | Record & monitor                           | Next quarterly review | —                  |

Mitigation examples:

| Hazard              | Typical fix                                        |
| ------------------- | -------------------------------------------------- |
| Bias disparity      | Re-sampling, fairness constraints, post-processing |
| Low accuracy        | Hyper-param tuning, more data, feature engineering |
| Drift vulnerability | Tighten monitoring thresholds, schedule re-train   |

---

## 4 Evidence & Record-Keeping

| Record            | Location                                           |
| ----------------- | -------------------------------------------------- |
| Risk register     | `compliance/risk_register.xlsx` (Modal volume)     |
| Markdown snapshot | `compliance/manual_fills/risk_register.md`         |
| Hazard JSON       | part of run metadata (`hazards` key)               |
| Jira tickets      | Project **AI-RISK** (auto-created for High/Medium) |

Retention ≥ 10 years (Art 12 policy).

---

## 5 Review Cycle

- **Per deployment** – Compliance Officer checks risk row before approving `approve_deployment` gate.
- **Quarterly** – Full review of open Medium / High risks (see `audit_plan.md`).
- **Post-incident** – Re-score affected model within 5 days.

---

_Last updated: **[YYYY-MM-DD]** – Next review due **[YYYY-MM-DD]**_
