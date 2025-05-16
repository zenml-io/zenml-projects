

# Annex IV: Technical Documentation for cs_deployment

## 1. General Description of the AI System

### 1(a) Intended Purpose and Version
**System Name:** cs_deployment<br>
**Provider:** ZenML GmbH<br>
**Description:** Demo Credit Scoring AI system<br>
**Pipeline Version:** 34725e01-65c2-4048-91c7-090ba6ff7995<br>
**Pipeline Run Version:** 1bfa5910-a05a-4df7-ad4f-d1c9bef5afcf


**Previous Versions:**<br>

1bfa5910-a05a-4df7-ad4f-d1c9bef5afcf, 

a7447a87-df50-4618-977e-6579eb072641, 

b60a698a-777d-4b93-86cd-636048077304, 

8af1f499-1e05-4ac3-ad50-94b7abe3a96b, 

54609dd3-55bf-444c-97cc-2e0c611337a8, 

e6e6d597-b9c2-48cb-804b-5a9aad99c146, 

07c7fb5f-34d7-48f8-af4f-2bd87e58a73a, 

77dc89cc-c02b-46d2-a8b1-9a981a9f359d, 

b668a71e-9e41-49cc-be93-c0355e9c0558, 

8106c508-a12f-4d00-a1e2-bf9e5b2867f4, 

4c805a6f-0206-44ec-b15d-04ef8178cde8, 

7cc38f65-6ae9-4d50-96cf-295045f91bf3, 

8473b791-eecd-4b74-8a4a-7cdf07796550, 

446cce90-f200-401c-a362-2eb181edc733



**Intended Purpose:** Assign a credit-risk score to loan applicants.

### 1(b) System Interactions

**Stack Components:** [REQUIRED: List the components of your ZenML stack]


**Additional Interactions:** Integrates with banking CRM systems for customer data extraction and loan decision workflows. The system also connects to credit bureau APIs to obtain applicants' credit history.


### 1(c) Software Versions

- Pipeline code commit: `7d16f4a30167c364cdc9b68e8a811ac572edbd62`


- Dependencies lock-file hash: [Not available]


**Framework Versions:**


- scikit-learn: 1.6.1



### 1(d) Deployment Forms
**Deployment Types:**

- [REQUIRED: Describe the forms in which this AI system is deployed (e.g., Docker image, REST API, Python SDK)]


### 1(e) Hardware Requirements
**Compute Resources:**

- CPUs: 
- Memory: 


### 1(f) Product Appearance

![Product](https://example.com/images/credit-scoring-system.png)


### 1(g) User Interface for Deployer

![Deployer UI](https://example.com/images/dashboard-screenshot.png)


### 1(h) Instructions for Use

**Documentation:** [User Guide](https://docs.example.com/credit-scoring-system)


## 2. Detailed Description of Elements and Development Process

### 2(a) Development Methods and Third-party Tools

**Pipeline Steps:** [REQUIRED: Describe the methods and steps performed for development]



**Source Code Repository:** [REQUIRED: Provide repository information]



**Integrations:** [REQUIRED: List third-party tools and integrations]


### 2(b) Design Specifications
**Model Architecture:** [REQUIRED: Specify model architecture]<br>
**Optimization Objective:** [REQUIRED: Specify what the system is designed to
optimize for]

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
****

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