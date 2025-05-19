

# Annex IV: Technical Documentation for cs_deployment

*Generated on 2025-05-18 22:05:21*

---

## 1. General Description of the AI System

### 1(a) Intended Purpose and Version

| **Field** | **Value** |
|-----------|-----------|
| **System Name** | cs_deployment |
| **Provider** | ZenML GmbH |
| **Description** | Demo Credit Scoring AI system |
| **Pipeline Version** | `34725e01-65c2-4048-91c7-090ba6ff7995` |
| **Pipeline Run Version** | `e7243682-a6f7-4f1d-b8ff-dc8da754994a` |


**Previous Versions:**

- **cs_deployment-2025_05_19-03_03_25_326264** (`9c54d912`) - 2025-05-19 03:03:26 *[failed]*

- **cs_deployment-2025_05_19-02_43_49_454817** (`c06273bc`) - 2025-05-19 02:43:50 *[completed]*

- **cs_deployment-2025_05_18-16_27_25_987810** (`f529a307`) - 2025-05-18 16:27:26 *[completed]*

- **cs_deployment-2025_05_18-16_16_24_116775** (`aa7360b3`) - 2025-05-18 16:16:25 *[completed]*

- **cs_deployment-2025_05_18-16_03_42_378610** (`7f02726b`) - 2025-05-18 16:03:43 *[completed]*

- **cs_deployment-2025_05_18-15_59_04_235495** (`e3056fd1`) - 2025-05-18 15:59:05 *[completed]*

- **cs_deployment-2025_05_18-15_58_11_396593** (`790ce633`) - 2025-05-18 15:58:12 *[failed]*

- **cs_deployment-2025_05_18-15_45_12_611631** (`955e15c3`) - 2025-05-18 15:45:13 *[completed]*

- **cs_deployment-2025_05_18-15_29_30_867372** (`a109c9ec`) - 2025-05-18 15:29:31 *[completed]*

- **cs_deployment-2025_05_18-15_28_29_739734** (`131d10f4`) - 2025-05-18 15:28:30 *[failed]*



**Intended Purpose:**
Assign a credit-risk score to loan applicants.

### 1(b) System Interactions


**ZenML Stack Configuration:**
- **Stack Name:** default
- **Stack ID:** `703cbbcf-5bd0-4746-8099-114bed3d07d9`

**Stack Components:**

**Artifact_store:**

- **default** (local, built-in)



**Orchestrator:**

- **default** (local, built-in)





**Additional System Interactions:**
Integrates with banking CRM systems for customer data extraction and loan decision workflows. The system also connects to credit bureau APIs to obtain applicants' credit history.

### 1(c) Software Versions


**Code Version Control:**
- **Pipeline Commit:** `1b9c2514963f1f4b959df8465ec2c60307d7fe4a`

- **Repository:** https://github.com/zenml-io/credit-scoring-ai-act.git





**Framework Versions:**

| Framework | Version |
|-----------|---------|
| datasets | >=3.6.0 |
| fairlearn | >=0.12.0 |
| flask | >=3.1.0 |
| pdfkit | >=1.0.0 |
| pyyaml | >=6.0.0 |
| markdown | >=3.8 |
| matplotlib | >=3.10.3 |
| modal | >=0.74.55 |
| openpyxl | >=3.1.5 |
| pandas | >=2.2.3 |
| plotly | >=6.0.1 |
| scikit-learn | >=1.6.1 |
| seaborn | >=0.13.2 |
| slack-sdk | >=3.35.0 |
| streamlit | >=1.45.1 |
| tabulate | >=0.9.0 |
| weasyprint | >=65.1 |
| whylogs | >=1.6.4 |
| xlsxwriter | >=3.2.3 |
| zenml | >=0.82.1 |


### 1(d) Deployment Forms

**Deployment Configuration:**
- **Type:** Modal + FastAPI (Serverless API deployment)
- **Environment:** Production
- **Scaling:** Automatic

### 1(e) Hardware Requirements

**Compute Resources:**

2 vCPU, 8 GB RAM, 10 GB disk


### 1(f) Product Appearance


![Product Overview](../../../assets/e2e.png)
*Figure 1: System Architecture Overview*


### 1(g) User Interface for Deployer


![Deployer Interface](../../../assets/streamlit-app.png)
*Figure 2: Deployment Interface*


### 1(h) Instructions for Use


**Documentation Resources:**
- [User Guide](../../../README.md)
- [API Documentation](api_guide.md)


---

## 2. Detailed Description of Elements and Development Process

### 2(a) Development Methods and Third-party Tools

**Pipeline Execution History:**


#### cs_feature_engineering
*Run ID: `d0b5c90c-781f-4d7d-a599-99620270aea0`*

| Step Name | Status | Inputs | Outputs |
|-----------|--------|---------|---------|
| **ingest** | ✅ completed | - | cs_data_profile=[2c29cf25], cs_whylogs_visualization=[6cec0a69], credit_scoring_df=[730acf8f] |
| **data_preprocessor** | ✅ completed | dataset_tst=9cacc62d, dataset_trn=db994304 | cs_test_df=[0bbae292], cs_preprocess_pipeline=[64efc0f7], cs_preprocessing_metadata=[836c1abb], cs_train_df=[950a6b55] |
| **data_splitter** | ✅ completed | dataset=730acf8f | raw_dataset_tst=[9cacc62d], raw_dataset_trn=[db994304] |
| **generate_compliance_metadata** | ✅ completed | test_df=0bbae292, data_profile=2c29cf25, preprocessing_metadata=836c1abb, train_df=950a6b55 | cs_compliance_record=[d872e5c7] |


#### cs_training
*Run ID: `80d5083a-00d0-4303-8126-8a0b0cc7ecc3`*

| Step Name | Status | Inputs | Outputs |
|-----------|--------|---------|---------|
| **train_model** | ✅ completed | test_df=0bbae292, train_df=950a6b55 | credit_scoring_model=[69410b29] |
| **evaluate_model** | ✅ completed | test_df=0bbae292, model=69410b29 | cs_evaluation_results=[0c9ee7e4] |
| **risk_assessment** | ✅ completed | evaluation_results=0c9ee7e4 | cs_risk_scores=[411d7faf] |


#### cs_deployment
*Run ID: `e7243682-a6f7-4f1d-b8ff-dc8da754994a`*

| Step Name | Status | Inputs | Outputs |
|-----------|--------|---------|---------|
| **approve_deployment** | ✅ completed | evaluation_results=0c9ee7e4, risk_scores=411d7faf | cs_approved=[080966f8], cs_approval_record=[68f85d04] |
| **generate_annex_iv_documentation** | 🔄 running | evaluation_results=0c9ee7e4, risk_scores=411d7faf | - |



**Development Environment:**
- **Source Code Repository:** https://github.com/zenml-io/credit-scoring-ai-act.git
- **Version Control System:** Git
- **CI/CD Platform:** ZenML Pipelines

### 2(b) Design Specifications

| **Specification** | **Details** |
|-------------------|-------------|
| **Model Architecture** | Gradient Boosting Decision Tree (XGBoost) |
| **Optimization Objective** | Maximize balanced accuracy while minimizing fairness disparities across protected groups |

**Design Rationale and Assumptions:**

We assume complete financial history, stable macroeconomy.


**Compliance Trade-offs:**

We reduced model complexity to improve explainability, which slightly reduced accuracy but significantly improved transparency in decision-making as required by the EU AI Act.


### 2(c) System Architecture


![System Architecture](../../../assets/e2e.png)
*Figure 3: Detailed System Architecture*


**Computational Resources:**

Training requires 8 vCPU, 32 GB RAM, and GPU acceleration. Inference runs on standard 2 vCPU, 8 GB RAM instances.


### 2(d) Data Requirements and Provenance


**Dataset Overview:**
- **Name:** Credit Scoring Dataset
- **Source:** Historical loan application data (5-year span)
- **Size:** ~10,000 records
- **Features:** Age, income, employment, credit history, debt, payment history
- **Target:** Binary credit risk classification


**Data Processing Methodology:**

Data is selected from a balanced historical loan dataset spanning 5 years. Sensitive attributes are removed or anonymized during preprocessing. Labels are derived from actual loan outcomes with a 24-month performance window.


### 2(e) Human Oversight Assessment


Human oversight is implemented through a mandatory approval workflow before any model deployment. The system flags potentially biased predictions for human review.


### 2(f) Predetermined Changes and Continuous Compliance


The system continuously monitors fairness metrics and performance. Automated retraining is triggered if drift is detected, with a quarterly schedule for full compliance reassessment.


### 2(g) Validation and Testing Procedures

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Accuracy | 0.87 |
| Precision | 0.82 |
| Recall | 0.79 |
| F1_score | 0.8 |
| Auc | 0.89 |
| Fairness_score | 0.92 |


**Fairness Assessment:**

| Fairness Metric | Score |
|-----------------|-------|
| Demographic Parity | 0.92 |
| Equal Opportunity | 0.89 |
| Equalized Odds | 0.85 |
| Disparate Impact Age | 0.95 |
| Disparate Impact Gender | 0.97 |


**Bias Mitigation Measures:**

We implemented post-processing techniques to minimize gender and age bias. Protected attributes are excluded from the model's training data, and we apply fairness constraints during prediction.


### 2(h) Cybersecurity Measures


The model and data are secured using access controls, encryption at rest and in transit, regular security scanning, and vulnerability assessment of the deployment infrastructure.


---

## 3. Monitoring, Functioning and Control

**System Capabilities and Limitations:**


- **Expected Accuracy:** 87.0%


**System Limitations:**

The system may have lower accuracy for applicants with limited credit history. It should not be used as the sole determining factor for loan approvals.


**Foreseeable Unintended Outcomes:**

The system may inadvertently perpetuate historical biases present in the training data. There is also a risk of creating a feedback loop where individuals denied credit cannot build credit history.


**Input Data Specifications:**

The system requires a minimum set of applicant data including income, employment history, existing debt, payment history, and credit score. All numerical inputs should be normalized and categorical variables encoded according to the documented schema.


---

## 4. Appropriateness of Performance Metrics


We selected AUC, precision, and recall as primary metrics based on the business need to balance risk with opportunity. Fairness metrics include demographic parity and equal opportunity difference to ensure equitable outcomes across protected groups.


---

## 5. Risk Management System


The risk management system follows a structured approach in accordance with Article 9 of the EU AI Act, consisting of:

1) Risk identification through cross-functional workshops and automated analysis
2) Risk assessment using a standardized scoring matrix for likelihood and impact
3) Risk mitigation with documented controls and responsible parties
4) Ongoing monitoring with automated drift detection
5) Regular reviews on a quarterly basis

The system prioritizes risks related to data quality, model fairness, security vulnerabilities, and performance degradation.


---

## 6. Lifecycle Changes Log


```
v1.0.0 (2025-03-15): Initial production model

v1.1.0 (2025-04-10): Updated preprocessing pipeline to improve handling of missing values

v1.2.0 (2025-05-01): Enhanced fairness constraints based on feedback from first month of operation

v1.2.1 (2025-05-15): Bug fix for edge case in feature normalization
```


---

## 7. Standards and Specifications Applied


ISO/IEC 27001:2022 for information security, IEEE 7010-2020 for algorithmic impact assessment, and CEN Workshop Agreement 17145-1 for validation methodologies.


---

## 8. EU Declaration of Conformity


```
EU Declaration of Conformity for Credit Scoring AI System

Provider: ZenML GmbH
Address: Example Street 123, 80331 Munich, Germany
Contact: compliance@zenml.io

We, ZenML GmbH, declare under our sole responsibility that the Credit Scoring AI System, version 1.2.1, complies with the relevant requirements set out in Section 2 of the EU AI Act.

The system has undergone conformity assessment in accordance with Article 43 and meets all requirements related to:
- Risk management (Article 9)
- Data governance (Article 10)
- Technical documentation (Article 11)
- Record keeping (Article 12)
- Transparency (Article 13)
- Human oversight (Article 14)
- Accuracy, robustness and cybersecurity (Article 15)

This declaration is kept at the disposal of national competent authorities for 10 years after the system has been placed on the market or put into service, in compliance with Article 47.

Signed for and on behalf of ZenML GmbH

Munich, May 15, 2025

John Doe
Chief Compliance Officer
```


---

## 9. Post-Market Monitoring Plan


We monitor model performance daily with automated alerting for drift detection. User feedback is collected through a structured feedback API and quarterly audits are conducted to assess overall system impact.


---

*End of Annex IV Documentation*

**Document Version:** e7243682-a6f7-4f1d-b8ff-dc8da754994a<br>
**Generated:** 2025-05-18 22:05:21<br>
**Compliance Standard:** EU AI Act (Regulation 2024/1689)