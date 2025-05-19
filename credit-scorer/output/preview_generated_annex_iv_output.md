

# Annex IV: Technical Documentation for Credit Scoring Model v1.2

## 1. General Description of the AI System

### 1(a) Intended Purpose and Version
**System Name:** Credit Scoring Model v1.2<br>
**Provider:** Example Corp.<br>
**Version:** 1.2.0
<br>
**Previous Versions:** 1.0.0, 1.1.0

**Intended Purpose:** AI system for evaluating loan applications

### 1(b) System Interactions

**Stack Components:** [REQUIRED: List the components of your ZenML stack]


**Additional Interactions:** [REQUIRED: Describe any interactions with
external systems (hardware, software, other AI systems)]

### 1(c) Software Versions

- Pipeline code commit: `abc123def456`


- Dependencies lock-file hash: [Not available]


**Framework Versions:**

[REQUIRED: List key frameworks and their versions]


### 1(d) Deployment Forms
**Deployment Types:**


- Docker container (2025-01-15)



### 1(e) Hardware Requirements
**Compute Resources:**

[REQUIRED: Describe hardware requirements for this AI system]


### 1(f) Product Appearance

[OPTIONAL: Attach photographs or illustrations showing external features of the product]


### 1(g) User Interface for Deployer

[REQUIRED: Provide a basic description of the user interface available to the deployer]


### 1(h) Instructions for Use

[REQUIRED: Provide instructions for use or link to documentation]


## 2. Detailed Description of Elements and Development Process

### 2(a) Development Methods and Third-party Tools

**Pipeline Steps:**

- Data preprocessing (transform): Clean and normalize input data




**Source Code Repository:** [REQUIRED: Provide repository information]



**Integrations:** [REQUIRED: List third-party tools and integrations]


### 2(b) Design Specifications
**Model Architecture:** XGBoost ensemble
**Optimization Objective:** F1 score with fairness constraints

**Key Design Rationale and Assumptions:**
  System assumes complete financial history is available

**Trade-offs Made for Compliance:**
  [REQUIRED: Describe any trade-offs made to comply with Chapter III,
  Section 2 of the EU AI Act]

### 2(c) System Architecture

[REQUIRED: Provide a description of the system architecture]


**Computational Resources:**
[REQUIRED: Describe computational resources used to develop, train,
test and validate]

### 2(d) Data Requirements and Provenance

[REQUIRED: Provide information about datasets used, including their provenance, scope, and characteristics]


**Data Processing Methodology:**
[REQUIRED: Describe data selection, labeling, and cleaning methodologies in
accordance with Article 10 of the EU AI Act]

### 2(e) Human Oversight Assessment
[REQUIRED: Provide an assessment of human oversight measures needed in
accordance with Article 14 of the EU AI Act, including technical measures to facilitate interpretation of outputs]

### 2(f) Predetermined Changes and Continuous Compliance
[REQUIRED: Describe any predetermined changes to the system and
technical solutions adopted to ensure continuous compliance with the EU AI Act]

### 2(g) Validation and Testing Procedures
**Performance Metrics:**

- Accuracy: 0.92
- AUC: 0.89


**Fairness Assessment:**

- Bias disparity: 0.05

- Demographic parity: 0.97


****

**Bias Mitigation Measures:**
Applied post-processing techniques to minimize gender and age bias

**Testing Documentation:**

**Test Logs:** [REQUIRED: Provide access to test logs and reports, dated and signed by responsible persons]


### 2(h) Cybersecurity Measures
[REQUIRED: Describe cybersecurity measures implemented to protect the AI
system]

## 3. Monitoring, Functioning and Control

**System Capabilities and Limitations:**

- Overall expected accuracy: 0.92

- System limitations: [REQUIRED: Describe system limitations, especially with
respect to specific persons or groups]

**Foreseeable Unintended Outcomes:**
[REQUIRED: Describe foreseeable unintended outcomes and sources of risks to
health, safety, fundamental rights, and discrimination in view of the intended purpose]

**Human Oversight Measures:**
[See section 2(e) for detailed human oversight assessment]

**Input Data Specifications:**
[REQUIRED: Specify requirements and characteristics for input data]

## 4. Appropriateness of Performance Metrics
[REQUIRED: Provide a description of why the chosen performance metrics
are appropriate for this specific AI system]

## 5. Risk Management System

[REQUIRED: Provide a detailed description of the risk management system in accordance with Article 9 of the EU AI Act]


## 6. Lifecycle Changes Log

[REQUIRED: Describe relevant changes made to the system through its lifecycle]


## 7. Standards and Specifications Applied
[REQUIRED: List harmonized standards applied in full or in part, or detailed
description of solutions adopted to meet requirements in Chapter III, Section 2]

## 8. EU Declaration of Conformity

[REQUIRED: Attach a copy of the EU declaration of conformity referred to in Article 47]


## 9. Post-Market Monitoring Plan
[REQUIRED: Provide a detailed description of the system in place to evaluate
AI system performance in the post-market phase in accordance with Article 72, including the post-market monitoring
plan]