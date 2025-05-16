# SOP – Data Ingestion & Preprocessing

_Version 0.1 • Owner: Data Governance Manager_

| Section               | Details                                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Ensure every dataset ingested into the feature engineering pipeline meets quality, lineage and compliance requirements (Art 10, 12).        |
| **Scope**             | All training/validation/test datasets used by `pipelines/feature_engineering.py`.                                                           |
| **Roles**             | Data/ML Eng. (executor) · Data Gov. Manager (approver).                                                                                     |
| **Inputs**            | HuggingFace dataset or raw CSV/Parquet files                                                                                                |
| **Outputs / Records** | • WhyLogs profile (`reports/data_profiles/`)<br>• `data_snapshot` metadata hash<br>• Preprocessing pipeline (Modal Volume + ZenML artifact) |

## 1. Data Loading

1. Configure source in pipeline config or env vars:
   - `HF_DATASET_NAME` (default = "spectrallabs/credit-scoring-training-dataset")
   - `TARGET_COLUMN` (default = "target")
2. **Automated checks** in `ingest` step:
   - SHA-256 hash calculation for provenance
   - WhyLogs profile generation
   - Sensitive attribute identification (based on `SENSITIVE_ATTRIBUTES` list)
   - Metadata logging to ZenML

## 2. Data Splitting

Always use stratified train-test split to maintain class distribution:

- Default 80/20 split ratio
- Uses `sklearn.model_selection.train_test_split`
- Logged as ZenML artifacts with schema

## 3. Protected Attributes

Protected attributes must be:

- Automatically identified based on `SENSITIVE_ATTRIBUTES` list in constants
- Preserved for fairness evaluation
- Documented in WhyLogs profile
- Tested for bias in `evaluate_model` step

## 4. Preprocessing Steps

| Step             | Implementation    | Configuration                                    |
| ---------------- | ----------------- | ------------------------------------------------ |
| Missing values   | Drop rows         | `drop_na=True/False`                             |
| Standardization  | StandardScaler    | `normalize=True/False`                           |
| Column dropping  | Explicit list     | `drop_columns` (`wallet_address` always dropped) |
| Pipeline storage | ColumnTransformer | Saved to Modal Volume                            |

## 5. Evidence Trail

| Artifact                | Location       | Step                           | Artifact/Metadata name   |
| ----------------------- | -------------- | ------------------------------ | ------------------------ |
| WhyLogs profile         | ZenML artifact | `ingest`                       | `cs_data_profile`        |
| SHA-256 hash & meta     | ZenML metadata | `ingest`                       | `data_snapshot`          |
| Compliance metadata     | ZenML metadata | `generate_compliance_metadata` | `cs_compliance_metadata` |
| Pre-processing pipeline | ZenML artifact | `data_preprocessor`            | `cs_preprocess_pipeline` |

## 6. Signs of Problematic Data

| Issue                    | Detection                | Response                |
| ------------------------ | ------------------------ | ----------------------- |
| Incomplete fields        | Missing > 5%             | Document in profile     |
| Class imbalance          | Target distribution skew | Use stratified split    |
| Outliers                 | Distribution analysis    | Report in metadata      |
| Protected attribute bias | Correlation analysis     | Flag for fairness tests |

---

_Last updated: [DATE]_  
_Approved by: [AI Compliance Officer]_
