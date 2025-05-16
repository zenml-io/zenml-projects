# Roles & Responsibilities – Credit-Scoring AI QMS

_Last updated: 2025-05-14_

| Role                          | Named Individual | Core Duties                                                 | Sign-off Authority          | Backup        |
| ----------------------------- | ---------------- | ----------------------------------------------------------- | --------------------------- | ------------- |
| **AI Compliance Officer**     | _TBD_            | Own QMS, final deployment approval, liaison with regulators | Deployment & risk overrides | CTO           |
| **ML Engineering Lead**       | _TBD_            | Maintain ZenML pipelines, model design & validation         | Code / pipeline merges      | Senior ML Eng |
| **Data Governance Manager**   | _TBD_            | Dataset quality & lineage, fairness features                | Dataset approval            | BI Lead       |
| **Quality Assurance Lead**    | _TBD_            | Test & validation protocols, metric verification            | Test sign-off               | QA Analyst    |
| **Incident Response Manager** | _TBD_            | Coordinate incident triage and RCA                          | Incident closure            | SRE Lead      |

_(Full descriptions live in `roles_and_responsibilities.md#appendix-a`)_

---

## Escalation Paths (Appendix A)

| Issue type             | First responder → Escalation 1 → Escalation 2    |
| ---------------------- | ------------------------------------------------ |
| **Model performance**  | ML Engineer → ML Lead → Compliance Officer       |
| **Data quality**       | Data Eng → Data Gov Manager → Compliance Officer |
| **Bias / fairness**    | ML Lead + Data Gov Manager → Compliance Officer  |
| **Regulatory inquiry** | Compliance Officer → Executive Sponsor           |

---

## Training Matrix (Appendix B)

Training records are maintained in the corporate LMS.  
Latest export: `appendices/training_matrix.pdf`.

| Role             | Mandatory modules                              | Refresh cycle    |
| ---------------- | ---------------------------------------------- | ---------------- |
| Core roles above | EU-AI-Act basics, Responsible AI, Incident SOP | Annual           |
| Engineers        | Fairlearn & drift monitoring deep dive         | Annual           |
| Managers         | Governance & sign-off workflow                 | Upon role change |

---

> **How to use:**
>
> 1. Replace `_TBD_` with real names.
> 2. Update the table whenever roles change (Git history is your audit trail).
> 3. Keep escalation and training info in the appendices to avoid bloating the main view.
