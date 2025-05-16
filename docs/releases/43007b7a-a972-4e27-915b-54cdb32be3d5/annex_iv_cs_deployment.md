# Annex IV: Technical Documentation for cs_deployment

## 1. General Description of the AI System

### 1(a) Intended Purpose and Version

**System Name:** cs_deployment<br>
**Provider:** <br>
**Description:** Demo Credit Scoring AI system<br>
**Pipeline Version:** 34725e01-65c2-4048-91c7-090ba6ff7995<br>
**Pipeline Run Version:** 43007b7a-a972-4e27-915b-54cdb32be3d5

**Previous Versions:** [None or not applicable]

**Intended Purpose:** Assign a credit-risk score to loan applicants.

### 1(b) System Interactions

**ZenML Stack:** local (ID: a51bf906-a221-46d7-81b8-c81f8adc5097)

**Stack Components:**

**Artifact_store:**

- local (local, built-in)

**Orchestrator:**

- local (local, built-in)

**Additional Interactions:** Integrates with banking CRM systems for customer data extraction and loan decision workflows. The system also connects to credit bureau APIs to obtain applicants' credit history.

### 1(c) Software Versions

- Pipeline code commit: `2e2599713dc5a6a54799503b692ecd6dfe5da85f`

- Dependencies lock-file hash: [Not available]

**Framework Versions:**

[REQUIRED: List key frameworks and their versions]

### 1(d) Deployment Forms

**Deployment Types:**

- Modal + FastAPI

### 1(e) Hardware Requirements

**Compute Resources:**

2 vCPU, 8 GB RAM, 10 GB disk

### 1(f) Product Appearance

![Product](https://example.com/images/credit-scoring-system.png)

### 1(g) User Interface for Deployer

![Deployer UI](https://example.com/images/dashboard-screenshot.png)

### 1(h) Instructions for Use

**Documentation:** [User Guide](https://docs.example.com/credit-scoring-system)

## 2. Detailed Description of Elements and Development Process

### 2(a) Development Methods and Third-party Tools

**Pipeline Steps:**

#### cs_feature_engineering (run 5500e6d0-5be2-4d0e-955b-ca9b72bb624f)

| Step name                    | Status    | Inputs                                                                                      | Outputs                                                                                                                |
| ---------------------------- | --------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| ingest                       | completed | -                                                                                           | credit_scoring_df=[21c48c8d], cs_whylogs_visualization=[4162cc6e], cs_data_profile=[d5744c7a]                          |
| data_preprocessor            | completed | dataset_trn=14dc1719, dataset_tst=6f2474a8                                                  | cs_test_df=[0d8ab2fb], cs_preprocessing_metadata=[4e52ed69], cs_train_df=[9a18d488], cs_preprocess_pipeline=[c845cb84] |
| data_splitter                | completed | dataset=21c48c8d                                                                            | raw_dataset_trn=[14dc1719], raw_dataset_tst=[6f2474a8]                                                                 |
| generate_compliance_metadata | completed | test_df=0d8ab2fb, preprocessing_metadata=4e52ed69, train_df=9a18d488, data_profile=d5744c7a | cs_compliance_metadata=[14b9d0a9]                                                                                      |

#### cs_training (run 4af71110-b951-44aa-978f-8b07fca405bc)

| Step name       | Status    | Inputs                              | Outputs                          |
| --------------- | --------- | ----------------------------------- | -------------------------------- |
| evaluate_model  | completed | test_df=0d8ab2fb, model=4e20aad2    | cs_evaluation_results=[cf1d8e72] |
| risk_assessment | completed | evaluation_results=cf1d8e72         | cs_risk_scores=[87edcc02]        |
| train_model     | completed | test_df=0d8ab2fb, train_df=9a18d488 | credit_scoring_model=[4e20aad2]  |

#### cs_deployment (run 43007b7a-a972-4e27-915b-54cdb32be3d5)

| Step name                       | Status    | Inputs                                            | Outputs                                               |
| ------------------------------- | --------- | ------------------------------------------------- | ----------------------------------------------------- |
| generate_annex_iv_documentation | running   | risk_scores=87edcc02, evaluation_results=cf1d8e72 | -                                                     |
| approve_deployment              | completed | risk_scores=87edcc02, evaluation_results=cf1d8e72 | cs_approved=[06039138], cs_approval_record=[a21e0240] |

**Source Code Repository:** [REQUIRED: Provide repository information]

**Integrations:** None

### 2(b) Design Specifications

**Model Architecture:** [REQUIRED: Specify model architecture]<br>
**Optimization Objective:** [REQUIRED: Specify what the system is designed to optimize for]

**Key Design Rationale and Assumptions:**
We assume complete financial history, stable macroeconomy.

**Trade-offs Made for Compliance:**
We reduced model complexity to improve explainability, which slightly reduced accuracy but significantly improved transparency in decision-making as required by the EU AI Act.

### 2(c) System Architecture

![Architecture Diagram](https://example.com/images/system-architecture.png)

**Computational Resources:**
Training requires 8 vCPU, 32 GB RAM, and GPU acceleration. Inference runs on standard 2 vCPU, 8 GB RAM instances.

### 2(d) Data Requirements and Provenance

[REQUIRED: Provide information about datasets used, including their provenance, scope, and characteristics]

**Data Processing Methodology:**
Data is selected from a balanced historical loan dataset spanning 5 years. Sensitive attributes are removed or anonymized during preprocessing. Labels are derived from actual loan outcomes with a 24-month performance window.

### 2(e) Human Oversight Assessment

Human oversight is implemented through a mandatory approval workflow before any model deployment. The system flags potentially biased predictions for human review.

### 2(f) Predetermined Changes and Continuous Compliance

The system continuously monitors fairness metrics and performance. Automated retraining is triggered if drift is detected, with a quarterly schedule for full compliance reassessment.

### 2(g) Validation and Testing Procedures

**Performance Metrics:**

[REQUIRED: List performance metrics used, including accuracy, robustness, and compliance metrics]

**Fairness Assessment:**

[REQUIRED: Provide comprehensive fairness and bias assessment metrics, including analysis across protected attributes]

**Bias Mitigation Measures:**
We implemented post-processing techniques to minimize gender and age bias. Protected attributes are excluded from the model's training data, and we apply fairness constraints during prediction.

**Testing Documentation:**

**Test Logs:** [REQUIRED: Provide access to test logs and reports, dated and signed by responsible persons]

### 2(h) Cybersecurity Measures

The model and data are secured using access controls, encryption at rest and in transit, regular security scanning, and vulnerability assessment of the deployment infrastructure.

## 3. Monitoring, Functioning and Control

**System Capabilities and Limitations:**

- Expected accuracy: [REQUIRED: Specify the expected level of accuracy]

- System limitations: The system may have lower accuracy for applicants with limited credit history. It should not be used as the sole determining factor for loan approvals.

**Foreseeable Unintended Outcomes:**
The system may inadvertently perpetuate historical biases present in the training data. There is also a risk of creating a feedback loop where individuals denied credit cannot build credit history.

**Human Oversight Measures:**
[See section 2(e) for detailed human oversight assessment]

**Input Data Specifications:**
The system requires a minimum set of applicant data including income, employment history, existing debt, payment history, and credit score. All numerical inputs should be normalized and categorical variables encoded according to the documented schema.

## 4. Appropriateness of Performance Metrics

We selected AUC, precision, and recall as primary metrics based on the business need to balance risk with opportunity. Fairness metrics include demographic parity and equal opportunity difference to ensure equitable outcomes across protected groups.

## 5. Risk Management System

[REQUIRED: Provide a detailed description of the risk management system in accordance with Article 9 of the EU AI Act]

## 6. Lifecycle Changes Log

[REQUIRED: Describe relevant changes made to the system through its lifecycle]

## 7. Standards and Specifications Applied

ISO/IEC 27001:2022 for information security, IEEE 7010-2020 for algorithmic impact assessment, and CEN Workshop Agreement 17145-1 for validation methodologies.

## 8. EU Declaration of Conformity

[EU Declaration of Conformity](https://compliance.example.com/eu-declaration-of-conformity.pdf)

## 9. Post-Market Monitoring Plan

We monitor model performance daily with automated alerting for drift detection. User feedback is collected through a structured feedback API and quarterly audits are conducted to assess overall system impact.
