# SOP – Incident Response

_Version: 0.1 • Owner: Incident Response Manager_

|             |                                                                            |
| ----------- | -------------------------------------------------------------------------- |
| Incident    | Event leading to harmful or discriminatory outcome or system outage > 4 h. |
| Legal basis | EU AI Act Art 18.                                                          |
| Roles       | First Responder, Incident Manager, Compliance Officer, Legal Advisor.      |
| Evidence    | `incident_log.json`, GitHub/Jira ticket, post-mortem doc.                  |

### Severity Classification

| Level              | Definition                              | Response Time | Notification     | External Report |
| ------------------ | --------------------------------------- | ------------- | ---------------- | --------------- |
| **A: Critical** 🚨 | Legal impact; harm to individuals       | Immediate     | All stakeholders | Required (72h)  |
| **B: High** 🔴     | Customer impact; significant bias       | < 4 hours     | Core team        | Optional        |
| **C: Medium** ⚠️   | Service degradation; performance issues | < 24 hours    | Team only        | No              |

### Process

1. Auto-alert from `report_incident.py` OR manual reporter opens Jira "AI-Incident".
2. **Severity classify** according to table above.
3. Notify core stakeholders via Slack `#credit-scoring-alerts` (template in `/templates/`).
4. Mitigate / rollback service.
5. Within 72 h: root-cause analysis and corrective-action plan.
6. Compliance Officer files external report to authority if severity A.
7. Close ticket when corrective actions verified.

### Documentation Requirements

- All incidents logged to `compliance/incident_log.json` via `incident_utils.py`
- Critical incidents (Level A) require formal post-mortem document
- Retain evidence (logs, metrics, user reports) linked to incident record
- Monthly incident summary reviewed by Compliance Officer

---

_Last updated: [DATE]_  
_Approved by: [AI Compliance Officer]_
