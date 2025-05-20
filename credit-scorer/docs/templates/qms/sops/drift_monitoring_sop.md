# SOP – Drift & Performance Monitoring

_Version 0.1 • Owner: ML Engineering Lead_

| Section               | Details                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| **Purpose**           | Detect and respond to data/concept drift to satisfy EU-AI-Act Art 17 (post-market monitoring). |
| **Scope**             | All deployed models in production environment.                                                 |
| **Roles**             | ML Engineer (executor) · Incident Manager (coordinator) · Compliance Officer (approver).       |
| **Inputs**            | Production data, baseline profiles, model predictions                                          |
| **Outputs / Records** | Monitoring reports, incident logs, mitigation actions                                          |

## 1. Automation Summary

| Frequency  | Pipeline / Job               | Key checks                   | Evidence                                                |
| ---------- | ---------------------------- | ---------------------------- | ------------------------------------------------------- |
| **Daily**  | `monitor_pipeline.py` (CRON) | • WhyLogs distance<br>• ΔAUC | JSON report → `compliance/monitoring/daily_<date>.json` |
| **Weekly** | Scheduled ZenML evaluation   | Full metrics vs baseline     | PDF → `reports/weekly_eval_<date>.pdf`                  |

Baseline profiles update automatically every time
`approve_deployment` approves a new model.

## 2. Thresholds (edit in `src/monitor_config.py`)

| Check               | Default threshold | Alert level |
| ------------------- | ----------------- | ----------- |
| WhyLogs JS-distance | > 0.15            | MEDIUM      |
| AUC drop            | > 0.05            | HIGH        |
| Fairness disparity  | > 0.20            | CRITICAL    |
| Prediction patterns | Shift > 15%       | MEDIUM      |

## 3. Response Flow

Alert → Incident Manager (Slack) → Jira "AI-Incident"
├─ MEDIUM: analyse within 5 d
├─ HIGH: RCA & plan in 48 h
└─ CRITICAL: consider model rollback

RCA template lives in `templates/incident_form.md`.
All incidents appended to `compliance/incident_log.json`.

## 4. Roles

| Role                   | Responsibilities                             |
| ---------------------- | -------------------------------------------- |
| **ML Engineer**        | Triage alerts, run RCA notebook              |
| **Incident Manager**   | Coordinate stakeholders, track resolution    |
| **Compliance Officer** | Approve mitigation actions or model rollback |

## 5. Documentation Requirements

All monitoring results must be preserved for compliance:

- Alert history with timestamps and response actions
- Investigation findings linked to each alert
- Mitigation actions and effectiveness assessment
- All stored in `compliance/monitoring/` and referenced in incident logs

## 6. Review & Improvement

- Drift thresholds reviewed **quarterly** (see `audit_plan.md`).
- SOP updated whenever metrics or tooling change.

---

_Last updated: [YYYY-MM-DD]_  
_Approved by: [AI Compliance Officer]_
