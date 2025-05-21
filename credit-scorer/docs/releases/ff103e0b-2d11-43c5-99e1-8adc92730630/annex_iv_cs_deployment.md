

# Annex IV: Technical Documentation for cs_deployment

*Generated on 2025-05-20 20:12:08*

---

## 1. General Description of the AI System

### 1(a) Intended Purpose and Version

| **Field** | **Value** |
|-----------|-----------|
| **System Name** | cs_deployment |
| **Provider** | ZenML GmbH |
| **Description** | EU AI Act Compliant Credit Scoring System for financial institutions |
| **Pipeline Version** | `34725e01-65c2-4048-91c7-090ba6ff7995` |
| **Pipeline Run Version** | `ff103e0b-2d11-43c5-99e1-8adc92730630` |

 **Previous Versions:**

|Version|Run ID|Created|Status|
|---|---|---|---|
|||||
|cs_deployment-2025_05_21-01_06_41_731637|`06dc37ff`|2025-05-21 01:06:42|completed|
|||||
|cs_deployment-2025_05_21-00_58_18_479813|`abd252ac`|2025-05-21 00:58:19|completed|
|||||
|cs_deployment-2025_05_21-00_45_24_844037|`e32fde78`|2025-05-21 00:45:25|completed|
|||||
|cs_deployment-2025_05_21-00_38_47_357032|`02f84060`|2025-05-21 00:38:48|completed|
|||||
|cs_deployment-2025_05_21-00_11_41_492947|`88538231`|2025-05-21 00:11:42|completed|
|||||
|cs_deployment-2025_05_20-22_36_28_080448|`99332e67`|2025-05-20 22:36:28|completed|
|||||
|cs_deployment-2025_05_20-21_52_00_462201|`21df97ba`|2025-05-20 21:52:01|failed|
|||||
|cs_deployment-2025_05_20-19_41_17_012665|`4ed3a819`|2025-05-20 19:41:17|failed|
|||||
|cs_deployment-2025_05_20-19_27_21_593920|`f0758de7`|2025-05-20 19:27:22|failed|
|||||
|cs_deployment-2025_05_20-18_34_39_514140|`de8d58f7`|2025-05-20 18:34:40|completed|
|||||
|||||

**Intended Purpose:**
To evaluate credit risk for loan applicants by providing an objective, fair, and transparent score based on financial history and demographic data.

### 1(b) System Interactions


**ZenML Stack Configuration:**
- **Stack Name:** local-s3
- **Stack ID:** `7a338bb7-c1b6-4c17-b2d4-6b5ca64c1723`

**Stack Components:**

**Orchestrator:**

- **default** (local, built-in)



**Artifact_store:**

- **aws-zenml-dev** (s3, s3)






**ZenML Stack Configuration:**

- **Stack Name:** local-s3
- **Stack ID:** `7a338bb7-c1b6-4c17-b2d4-6b5ca64c1723` 
- **Created:** 2025-02-24 13:52:30  
- **Updated:** 2025-02-24 13:52:30  

**Stack Components:**



#### Orchestrator Components

|Name|Flavor|Integration|Component ID|
|---|---|---|---|
|||||
|default|local|built-in|`749034d4`|
|||||



#### Artifact_store Components

|Name|Flavor|Integration|Component ID|
|---|---|---|---|
|||||
|aws-zenml-dev|s3|s3|`f4d2a196`|
|||||




**Additional System Interactions:**
The system processes applicant data through a standardized input format and provides credit risk scores through a REST API. It integrates with ZenML for model versioning and deployment management, and Modal for serverless deployment. The system maintains a risk register and incident log for compliance tracking.

### 1(c) Software Versions


**Code Version Control:**
- **Pipeline Commit:** `6bb6c920c44f68a39394c530df6f6536df9c77de`

- **Repository:** https://github.com/zenml-io/zenml-projects/





**Framework Versions:**

| Framework | Version |
|-----------|---------|
| cyclonedx-python-lib | >=10.0.1 |
| fairlearn | >=0.12.0 |
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
| streamlit-option-menu | >=0.4.0 |
| tabulate | >=0.9.0 |
| weasyprint | >=65.1 |
| whylogs | latest |
| xlsxwriter | >=3.2.3 |
| zenml | >=0.82.1 |


### 1(d) Deployment Forms

**Deployment Configuration:**

- **Type:** Modal + FastAPI (Serverless API deployment with auto-scaling)
- **Environment:** Production
- **Scaling:** Automatic   

||

### 1(e) Hardware Requirements

**Compute Resources:**

Standard deployment: 2 vCPU, 1 GB RAM, 10GB disk


### 1(f) Product Appearance


![Product Overview](../../../assets/e2e.png)
*Figure 1: System Architecture Overview*


### 1(g) User Interface for Deployer


![Deployer Interface](../../../assets/streamlit-app.png)
*Figure 2: Deployment Interface*


### 1(h) Instructions for Use


**Documentation Resources:**
- [User Guide](../../../README.md)
- [API Documentation](../../../modal_app/api_guide.md)


---

## 2. Detailed Description of Elements and Development Process

### 2(a) Development Methods and Third-party Tools

**Pipeline Execution History:**


#### cs_feature_engineering
*Run ID: `b53c6ddc-0b2b-4b22-86d1-f6f3382c1e93`*

| Step Name | Status | Inputs | Outputs |
|-----------|--------|---------|---------|
| **data_preprocessor** | ✅ completed | dataset_tst=`2ecb8d86`, dataset_trn=`da966a4d` | cs_preprocess_pipeline=[`1ca0d40c`], cs_train_df=[`c0cc2858`], cs_test_df=[`e0a5e631`] |
| **data_splitter** | ✅ completed | dataset=`7094e22b` | raw_dataset_tst=[`2ecb8d86`], raw_dataset_trn=[`da966a4d`] |
| **ingest** | ✅ completed | - | cs_whylogs_visualization=[`217daa35`], credit_scoring_df=[`7094e22b`] |


#### cs_training
*Run ID: `ee45a707-5150-4ffa-9712-7ab5c85acfb5`*

| Step Name | Status | Inputs | Outputs |
|-----------|--------|---------|---------|
| **risk_assessment** | ✅ completed | evaluation_results=`b48febd8` | cs_risk_scores=[`41b49935`] |
| **evaluate_model** | ✅ completed | model=`5aaedd84`, test_df=`e0a5e631` | cs_evaluation_results=[`b48febd8`], cs_evaluation_visualization=[`b94178d1`] |
| **train_model** | ✅ completed | train_df=`c0cc2858`, test_df=`e0a5e631` | credit_scoring_model=[`5aaedd84`] |


#### cs_deployment
*Run ID: `ff103e0b-2d11-43c5-99e1-8adc92730630`*

| Step Name | Status | Inputs | Outputs |
|-----------|--------|---------|---------|
| **generate_sbom** | ✅ completed | - | cs_sbom_artifact=[`44b8f8b1`] |
| **approve_deployment** | ✅ completed | risk_scores=`41b49935`, evaluation_results=`b48febd8` | cs_approval_record=[`033db4e2`], cs_approved=[`2d36f1be`] |
| **generate_annex_iv_documentation** | 🔄 running | risk_scores=`41b49935`, evaluation_results=`b48febd8`, deployment_info=`f8706875` | - |
| **modal_deployment** | ✅ completed | approval_record=`033db4e2`, preprocess_pipeline=`1ca0d40c`, approved=`2d36f1be`, model=`5aaedd84`, evaluation_results=`b48febd8` | cs_deployment_info=[`f8706875`] |



**Development Environment:**
- **Source Code Repository:** https://github.com/zenml-io/zenml-projects/
- **Version Control System:** Git
- **CI/CD Platform:** ZenML Pipelines

### 2(b) Design Specifications

| **Specification** | **Details** |
|-------------------|-------------|
| **Model Architecture** | Gradient Boosting Decision Tree (XGBoost) |
| **Optimization Objective** | Maximize balanced accuracy while minimizing fairness disparities across protected demographic groups |

**Design Rationale and Assumptions:**

The model assumes applicants have a reasonably complete financial history and operates under stable macroeconomic conditions. It also assumes data quality standards are maintained across input sources.


**Compliance Trade-offs:**

To ensure EU AI Act compliance, we prioritized model explainability and fairness over maximum predictive performance. We implemented post-processing fairness constraints and transparent feature importance, which slightly reduced raw accuracy but significantly improved demographic fairness and transparency.


### 2(c) System Architecture


![System Architecture](../../../assets/modal-deployment.png)
*Figure 3: Detailed System Architecture*


**Computational Resources:**

Training infrastructure: 2 vCPU, 1 GB RAM. Inference runs on standard 2 vCPU, 1 GB RAM instances with auto-scaling based on demand. ZenML orchestration requires minimal additional resources.


### 2(d) Data Requirements and Provenance


**Dataset Overview:**
- **Name:** Credit Scoring Dataset
- **Source:** Historical loan application data (5-year span)
- **Size:** ~10,000 records
- **Features:** Age, income, employment, credit history, debt, payment history
- **Target:** Binary credit risk classification


**Data Processing Methodology:**

Training data is derived from a balanced historical loan dataset spanning 5 years with approximately 20,000 records. Sensitive attributes (gender, age) are protected during model training with fairness constraints. Feature engineering follows established credit risk assessment standards, with data cleaning and normalization protocols to handle outliers and missing values.


### 2(e) Human Oversight Assessment


Human oversight is implemented through a mandatory approval workflow before any deployment. The system's risk score output includes confidence intervals and feature importance to assist human reviewers. High-risk cases and those with potential bias are automatically flagged for human review, with specific thresholds established for different demographic groups.


### 2(f) Predetermined Changes and Continuous Compliance


The continuous compliance framework includes: (1) Daily automated monitoring of model drift metrics, (2) Weekly fairness audits across protected attributes, (3) Monthly performance reviews of model outputs, (4) Quarterly comprehensive compliance reassessments with stakeholder reviews, and (5) Incident response protocols for any detected bias or performance issues.


### 2(g) Validation and Testing Procedures

**Performance Metrics:**

| Metric | Value |
|--------|-------|
| Accuracy | None |
| Auc | None |


**Fairness Assessment:**

| Fairness Metric | Score |
|-----------------|-------|
| Demographic Parity Gender | None |
| Equal Opportunity Gender | None |
| Accuracy Disparity Gender | None |
| Selection Rate Disparity Gender | None |
| Accuracy Disparity Education | None |
| Selection Rate Disparity Education | None |
| Accuracy Disparity Housing | None |
| Selection Rate Disparity Housing | None |
| Overall Fairness Score | None |


**Bias Mitigation Measures:**

Our comprehensive bias mitigation approach includes: (1) Balanced dataset sampling for training, (2) Regular bias audits across protected attributes including gender, age, education level, family status, and housing type, (3) Post-processing fairness adjustments to equalize performance across demographic groups, and (4) Continuous monitoring of selection rate disparities for real-time fairness assessment.


### 2(h) Cybersecurity Measures


Security measures include: (1) End-to-end encryption for all data in transit and at rest, (2) Access controls with role-based permissions for model interaction, (3) Audit logging of all system access and predictions, (4) Regular vulnerability assessments of the deployment infrastructure, (5) Secure CI/CD pipeline with automated security scanning, and (6) Protection against model extraction or adversarial attacks.


---

## 3. Monitoring, Functioning and Control

**System Capabilities and Limitations:**


- **Expected Accuracy:** *[REQUIRED: Specify expected accuracy level]*


**System Limitations:**

The system has verified limitations including: (1) Lower accuracy for applicants with limited credit history, (2) Potential for reduced performance during significant macroeconomic shifts, (3) Reduced explainability for edge cases, and (4) Applicability only within the regulatory jurisdiction it was trained for. The model should always be used as a decision support tool, not as the sole determining factor for credit decisions.


**Foreseeable Unintended Outcomes:**

Potential unintended outcomes include: (1) Risk of perpetuating historical biases present in training data despite mitigation measures, (2) Possible feedback loops where denied applicants cannot build sufficient credit history for future approval, (3) Over-reliance on algorithmic decisions by human reviewers (automation bias), and (4) Potential for emerging disparities across intersectional demographic groups not specifically monitored.


**Input Data Specifications:**

Required input data includes: (1) Financial history (income, debt-to-income ratio, existing credit), (2) Employment data (job stability, industry sector), (3) Credit bureau information (credit score, previous defaults), (4) Payment history (timeliness, reliability), and (5) Demographic information (used only for fairness assessment). All numerical inputs must be normalized according to the documented preprocessing pipeline.


---

## 4. Appropriateness of Performance Metrics


The selected metrics provide a balanced assessment of model performance and fairness: (1) Accuracy (91.9%) measures overall predictive capability, (2) AUC (0.75) assesses discrimination ability across thresholds, (3) Selection rate disparities across protected groups quantify fairness, and (4) Per-group accuracy measures ensure consistent performance. These metrics align with both regulatory requirements and business objectives of fair lending.


---

## 5. Risk Management System


Our comprehensive risk management system implements Article 9 requirements through:

1) Risk Identification: Cross-functional workshops with stakeholders identify potential risks across technical performance, fairness, data quality, and operational domains.

2) Risk Assessment: Standardized scoring matrix evaluates likelihood and impact of each risk, with special attention to high-risk bias factors with an overall risk score of 0.525.

3) Risk Mitigation: Each identified risk has documented controls and responsible parties, with specific focus on bias mitigation achieving a risk score reduction of approximately 20%.

4) Continuous Monitoring: Automated drift detection alerts stakeholders to potential performance issues, with thresholds calibrated to regulatory requirements.

5) Regular Review: Quarterly reviews evaluate risk control effectiveness, with particular attention to risk measures of accuracy (risk score 0.25) and bias (risk score 0.8).

The system prioritizes monitoring and mitigating risks related to data quality, model fairness, security vulnerabilities, and performance degradation.


---

## 6. Lifecycle Changes Log


```
v1.0.0 (2025-03-01): Initial production model with baseline fairness constraints

v1.1.0 (2025-03-15): Enhanced preprocessing pipeline for improved missing value handling

v1.2.0 (2025-04-10): Implemented post-processing fairness adjustments based on initial performance data

v1.2.1 (2025-04-25): Optimized feature engineering process for protected attributes

v1.3.0 (2025-05-18): Comprehensive update with improved bias mitigation and EU AI Act compliance documentation
```


---

## 7. Standards and Specifications Applied


The system adheres to: (1) ISO/IEC 27001:2022 for information security management, (2) IEEE 7010-2020 for wellbeing impact assessment, (3) ISO/IEC 25024:2015 for data quality, (4) CEN Workshop Agreement 17145-1 for validation methodologies in AI, and (5) ISO/IEC 29119 for software testing.


---

## 8. EU Declaration of Conformity


```
EU Declaration of Conformity for Credit Scoring AI System

Provider: ZenML GmbH
Address: Example Street 123, 80331 Munich, Germany
Contact: compliance@zenml.io

We, ZenML GmbH, declare under our sole responsibility that the Credit Scoring AI System, version 1.3.0, complies with the relevant requirements set out in Section 2 of the EU AI Act (Regulation 2024/1689).

The system has undergone conformity assessment in accordance with Article 43 and meets all requirements related to:
- Risk management (Article 9)
- Data governance (Article 10)
- Technical documentation (Article 11)
- Record keeping (Article 12)
- Transparency (Article 13)
- Human oversight (Article 14)
- Accuracy, robustness and cybersecurity (Article 15)
- Post-market monitoring (Articles 16-17)
- Incident reporting (Articles 18-19)

This declaration is kept at the disposal of national competent authorities for 10 years after the system has been placed on the market or put into service, in compliance with Article 47.

Signed for and on behalf of ZenML GmbH

Munich, May 18, 2025

Louisa Gerryts
Chief Compliance Officer
```


---

## 9. Post-Market Monitoring Plan


The comprehensive post-market monitoring includes: (1) Daily automated performance monitoring with statistical drift detection, (2) Weekly fairness audits across protected attributes, (3) Structured feedback collection through API endpoints for user experience, (4) Quarterly compliance reassessment with stakeholder review, and (5) Incident tracking and response protocols for any detected issues or complaints.


---

*End of Annex IV Documentation*

**Document Version:** ff103e0b-2d11-43c5-99e1-8adc92730630

**Generated:** 2025-05-20 20:12:08

**Compliance Standard:** EU AI Act (Regulation 2024/1689)