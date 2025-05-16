# Internal-Audit & Management-Review Plan

_Credit-Scoring AI QMS • 2025-2026_

---

## 1 Objectives

- Verify compliance with EU-AI-Act Articles 9-18
- Check that automated evidence (Annex IV PDFs, risk register, fairness logs)
  is complete and accurate
- Identify process gaps and drive continual improvement

## 2 Audit Schedule (rolling)

| Quarter     | Window    | Focus                            | Lead Auditor               | Key Evidence                               | Status    |
| ----------- | --------- | -------------------------------- | -------------------------- | ------------------------------------------ | --------- |
| **Q3 2025** | 15–19 Sep | Data lineage & ingest SOP        | QA Lead                    | `data_profiles/`, `data_snapshot` metadata | _Planned_ |
| **Q4 2025** | 10–14 Dec | Fairness & risk scoring          | Compliance Officer         | `fairness/*.json`, `risk_register.xlsx`    | _Planned_ |
| **Q1 2026** | 10–14 Mar | Incident log & monitoring alerts | Incident Manager           | `incident_log.json`, drift metrics         | _Planned_ |
| **Q2 2026** | 12–16 Jun | Full QMS doc-control audit       | External ISO-42001 auditor | Git history of `compliance/qms/`           | _Planned_ |

_Edit dates / add rows as needed. Urgent “post-incident” audits can be added ad-hoc._

## 3 Scope & Evidence Map

| Area                       | Evidence to sample                             | SOP / File                                  |
| -------------------------- | ---------------------------------------------- | ------------------------------------------- |
| **Risk Management**        | `compliance/risk_register.xlsx`                | `sops/risk_mitigation_sop.md`               |
| **Model Dev & Validation** | Annex IV PDFs in `reports/`                    | DAG: `pipelines/credit_scoring_pipeline.py` |
| **Fairness & Bias**        | Summaries in run metadata (`fairness_summary`) | `evaluate_model` step                       |
| **Monitoring & Incidents** | `incident_log.json`, monitoring alerts         | `sops/incident_reporting_sop.md`            |
| **Documentation Control**  | Git commit history; file checksums             | `qms_manual.md` §11                         |

## 4 Method (condensed)

1. **Pre-audit:** lead auditor pulls latest evidence, reviews previous findings.
2. **Execution:**
   - Document review (Annex IV, risk register, incident log)
   - Spot-check ZenML run metadata (via dashboard or `zenml run describe`)
   - Interviews (roles listed in `roles_and_responsibilities.md`)
3. **Reporting:** use `templates/audit_report_template.md`.
4. **Follow-up:** corrective actions are filed as Jira tickets and
   tracked until closure; verification in next audit.

## 5 KPIs & Acceptance Criteria

| KPI                                          | Target |
| -------------------------------------------- | ------ |
| % runs with completed Annex IV               | 100 %  |
| Open high-severity incidents older than 30 d | 0      |
| Risk register rows without mitigation        | < 5 %  |
| Fairness disparity > 0.2 un-mitigated        | 0      |

_(KPIs are pulled automatically via `scripts/kpi_dashboard.py`; adjust thresholds if business context changes.)_

## 6 Management Review

- **Frequency:** quarterly (calendar invite “AI QMS Review”).
- **Inputs:** latest audit reports, KPI dashboard, incident summary.
- **Outputs:** management-review minutes filed at
  `compliance/qms/review_minutes/YYYY_QX.md` and action items in Jira.

---

_Last updated: **[YYYY-MM-DD]**_  
_Approved by:_ **[AI Compliance Officer Name]**
