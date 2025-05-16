# Credit‑Scoring EU AI Act Demo

> A ZenML‑powered end‑to‑end credit‑scoring workflow that automatically generates the technical evidence required by the EU AI Act.

## 🚀 Project Overview

The project implements three main pipelines:

1. **Feature Engineering**: Handles data governance and preprocessing (Articles 10, 12, 15)
   – `ingest → data_splitter → data_preprocessor → generate_compliance_metadata`

2. **Training Pipeline**: Implements model training, evaluation, and risk assessment (Articles 9, 11, 15)  
   – `train_model → evaluate_model → risk_assessment`

3. **Deployment Pipeline**: Manages human oversight, deployment, and monitoring (Articles 14, 17, 18)
   – `approve_deployment → modal_deployment → post_market_monitoring → post_run_annex`

Each run automatically versions its inputs, logs hashes & metrics, and generates a complete Annex IV draft.

## Architecture

![End-to-End Architecture](docs/e2e.png)

## Project Structure

```bash
credit_scoring_ai_act/
├── app/ # Modal deployment app
│   ├── modal_deployment.py # Modal deployment script
│   └── schemas.py # Pydantic models
├── src/
│   ├── pipelines/
│   │   ├── feature_engineering.py # Feature engineering pipeline
│   │   ├── training.py # Model training pipeline
│   │   └── deployment.py # Deployment pipeline
│   ├── steps/
│   │   ├── ingest.py # Load CSV → log SHA‑256, WhyLogs profile
│   │   ├── data_preprocessor.py # Basic feature engineering
│   │   ├── data_splitter.py # Split dataset into train/test
│   │   ├── generate_compliance_metadata.py # Generate compliance metadata
│   │   ├── train.py # XGBoost / sklearn model
│   │   ├── evaluate.py # Standard + Fairness metrics
│   │   ├── approve.py # Human‑in‑loop gate (approve_deployment step)
│   │   ├── post_market_monitoring.py # Post‑market monitoring
│   │   ├── generate_sbom.py # Generate SBOM
│   │   ├── post_run_annex.py # Generate Annex IV documentation
│   │   ├── risk_assessment.py # Risk assessment
│   │   └── deploy.py # Push to Modal / local FastAPI
│   ├── utils/ # Shared utilities
│   │   ├── modal_utils.py # Modal Volume operations
│   │   ├── preprocess.py # Custom sklearn transformers
│   │   ├── eval.py # Evaluation utils
│   │   ├── incidents.py # Incident reporting system
│   │   ├── visualizations.py # Visualization utils
│   │   └── model_definition.py # ZenML model definition
│   │
│   ├── configs/ # Configuration files
│   └── constants.py # Centralized configuration constants
│
├── docs/
│   ├── risk/ # Auto‑generated annex iv reports after deployment
│   ├── releases/ # Manual compliance inputs organized by run ID
│   │   ├── 07c7fb5f-34d7-48f8-af4f-2bd87e58a73a/  # Example run ID
│   │   │   ├── annex_iv_cs_deployment.md  # Annex IV deployment documentation
│   │   │   ├── evaluation_results.yaml    # Model evaluation metrics
│   │   │   ├── git_info.md                # Git repository information
│   │   │   ├── missing_fields.txt         # Fields missing from documentation
│   │   │   ├── README.md                  # Release-specific information
│   │   │   ├── risk_scores.yaml           # Risk assessment scores
│   │   │   └── sbom.json                  # Software Bill of Materials
│   │   └── e6e6d597-b9c2-48cb-804b-5a9aad99c146/  # Another run ID
│   └── templates/
│       ├── annex_iv_template.j2 # Annex IV template
│       ├── sample_inputs.json # Sample inputs for Annex IV
│       └── qms/ # Quality management system documentation
│           ├── qms_template.md # Core QMS document
│           ├── roles_and_responsibilities.md # Role assignments
│           ├── audit_plan.md # Audit schedule and methodology
│           └── sops/ # Standard Operating Procedures
│               ├── model_release_sop.md # Model release protocol
│               ├── drift_monitoring_sop.md # Monitoring procedures
│               ├── incident_response_sop.md # Incident handling
│               ├── risk_mitigation_sop.md # Risk management process
│               └── data_ingestion_sop.md # Data handling procedures
│
├── assets/ # Pipeline diagrams
├── run.py # CLI entrypoint
└── README.md
```

## Running Pipelines

```bash
# Run feature engineering pipeline (Articles 10, 12)
python run.py --feature

# Run model training pipeline (Articles 9, 11, 15)
python run.py --train

# Run deployment pipeline (Articles 14, 17, 18)
python run.py --deploy
```

Options:

- `--auto-approve` for non‑interactive deployment
- `--no-cache` to disable ZenML caching
- `--config-dir <path>` to override default configs

### Configuration

Pipeline configurations are stored in the `src/configs/` directory:

- `feature_engineering.yaml`
- `training.yaml`
- `deployment.yaml`

You can specify a custom config directory using the `--config-dir` option.

## Modal Deployment

![Modal Deployment](docs/modal-deployment.png)

The project implements a serverless deployment using Modal with comprehensive monitoring and incident reporting capabilities:

- FastAPI application with documented endpoints
- Automated model and preprocessing pipeline loading
- Drift detection and incident reporting
- Standardized storage paths for compliance artifacts

## 🔗 EU AI Act Compliance Mapping

For a complete overview of the EU AI Act compliance mapping, refer to the [detailed pipeline steps to articles mapping and interdependencies for full compliance](COMPLIANCE.md).

## Compliance Directory Structure

| Directory               | Purpose                                                 | Auto/Manual |
| ----------------------- | ------------------------------------------------------- | ----------- |
| **records/**            | Automated compliance records from pipeline runs         | Auto        |
| **manual_fills/**       | Manual compliance inputs and preprocessing info         | Manual      |
| **monitoring/**         | Post-market monitoring records and drift detection logs | Auto        |
| **deployment_records/** | Model deployment history and model cards                | Auto        |
| **approval_records/**   | Human approval records and rationales                   | Manual      |
| **templates/**          | Jinja template for Annex IV document generation         | Manual      |

## 📋 Quality Management System (Article  17)

This repo delivers all _technical evidence_ (lineage, metadata, logs). For a complete QMS, you must also maintain formal QMS documentation.

See `compliance/qms/` for starter templates:

- **Quality Policy** (`qms_template.md`)
- **Roles & Responsibilities** (`roles_and_responsibilities.md`)
- **Audit Plan** (`audit_plan.md`)
- **SOPs** (`sops/` folder: data ingestion, model release, risk mitigation, etc.)

## Docs

For detailed explanations of each pipeline and step, refer to the [detailed pipeline documentation](docs/detailed_pipeline_explanations.md).
