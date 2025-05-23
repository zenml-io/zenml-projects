## Interdependencies for Full Compliance

This matrix demonstrates how the pipeline achieves comprehensive EU AI Act compliance through distributed responsibility across steps:

| Article                              | Primary Steps                            | Supporting Steps                                   | Complete When                                |
| ------------------------------------ | ---------------------------------------- | -------------------------------------------------- | -------------------------------------------- |
| **Art. 9** (Risk Management)         | risk_assessment                          | ingest, evaluate_model, post_market_monitoring     | Risk register complete with mitigations      |
| **Art. 10** (Data Governance)        | ingest, data_preprocessor                | evaluate_model, generate_sbom                      | Data quality profiled and documented         |
| **Art. 11** (Technical Docs)         | generate_annex_iv_documentation          | ingest, train_model, evaluate_model, generate_sbom | Annex IV documentation generated             |
| **Art. 12** (Record-keeping)         | ingest                                   | All other steps                                    | Comprehensive logging throughout pipeline    |
| **Art. 13** (Transparency)           | modal_deployment                         | evaluate_model, risk_assessment                    | Model card and user information complete     |
| **Art. 14** (Human Oversight)        | approve_deployment                       | ingest, evaluate_model, risk_assessment            | Human review with approval record            |
| **Art. 15** (Accuracy & Robustness)  | evaluate_model, generate_sbom            | risk_assessment                                    | Performance metrics documented and evaluated |
| **Art. 16** (Quality Management)     | approve_deployment, modal_deployment     | All steps                                          | Quality management system implemented        |
| **Art. 17** (Post-market Monitoring) | post_market_monitoring, modal_deployment | ingest, evaluate_model                             | Monitoring plan implemented with thresholds  |
| **Art. 18** (Incident Notification)  | modal_deployment                         | post_market_monitoring                             | Incident reporting mechanism established     |
| **Art. 28** (Provider Obligations)   | generate_annex_iv_documentation          | ingest, train_model, modal_deployment              | All compliance documentation available       |
