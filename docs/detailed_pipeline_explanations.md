# Pipeline Architecture Analysis

## 1. DETAILED PIPELINE EXPLANATIONS

### Feature Engineering Pipeline

The Feature Engineering Pipeline forms the foundation of your credit scoring system, focused on data preparation while ensuring EU AI Act compliance.

#### Step-by-Step Process:

1. **Data Loading (Article 10 Compliance):**

   - Sources data from HuggingFace (specifically from HF_DATASET_NAME)
   - Calculates SHA-256 hash for data provenance and traceability
   - Creates WhyLogs profile for data quality documentation
   - Identifies potential sensitive attributes automatically
   - Logs comprehensive metadata to ZenML for lineage tracking
   - Samples data if requested (using stratified sampling to maintain class distribution)

2. **Data Splitting:**

   - Implements a train-test split using scikit-learn
   - Preserves dataset schema and applies stratification
   - Returns clearly named raw training and test datasets

3. **Data Preprocessing (Article 10 Compliance):**

   - Handles missing values by dropping NA values if requested
   - Removes specified columns (automatic removal of sensitive data like wallet addresses)
   - Standardizes numerical features using StandardScaler if normalization is requested
   - Creates and fits a preprocessing pipeline using scikit-learn's ColumnTransformer
   - Saves preprocessing pipeline to Modal Volume for deployment
   - Logs checksums and transformation details to ensure data providence

4. **Compliance Metadata Generation (Articles 10, 12, 15):**
   - Documents feature specifications, types, and statistics
   - Identifies and flags sensitive attributes for fairness checks
   - Tracks data quality metrics before and after preprocessing
   - Creates comprehensive compliance record for documentation
   - Saves artifacts to persistent storage for auditing purposes

#### Data Flow:

- Raw data from HuggingFace → Data loader → Data splitter → Preprocessor → Final training/test splits
- Metadata and artifacts flow to both ZenML tracking and Modal Volume storage
- Compliance documentation is generated and stored in local directories

#### Storage Operations:

- WhyLogs profiles saved to compliance/data_profiles/
- Preprocessing pipeline saved to Modal Volume at volume_metadata["preprocess_pipeline_path"]
- Compliance metadata logged to ZenML via log_metadata()

### Training Pipeline

The Training Pipeline implements model training with comprehensive fairness evaluation and risk assessment mechanisms to ensure EU AI Act compliance.

#### Step-by-Step Process:

1. **Model Training (Article 11 Compliance):**

   - Supports training with either pipeline-supplied data or retrieval from ZenML artifacts
   - Trains a GradientBoostingClassifier with configurable hyperparameters
   - Handles preprocessed feature names with transformation suffixes
   - Records training parameters, start time, and duration
   - Saves model locally (models/model.pkl) and to Modal Volume
   - Creates a model card with purpose, limitations, and performance metrics
   - Logs model checksums for integrity verification

2. **Model Evaluation (Articles 9, 15 Compliance):**

   - Calculates performance metrics (accuracy, AUC)
   - Performs fairness analysis using Fairlearn's MetricFrame
   - Checks selection rates and accuracy across protected attributes
   - Flags bias if selection rate disparity exceeds threshold (0.2)
   - Captures comprehensive fairness metrics per demographic group
   - Saves fairness report to Modal Volume for compliance documentation
   - Optional integration point for Slack alerts on bias detection

3. **Risk Assessment (Article 9 Compliance):**
   - Processes evaluation results to calculate risk scores
   - Updates risk register in Excel format with quantified risk metrics
   - Creates a Markdown snapshot of the risk register
   - Categorizes risk level and suggests mitigations
   - Logs risk assessment results to ZenML

#### Data Flow:

- Training/test datasets → Model training → Model path → Evaluation
- Evaluation results → Risk assessment
- Metadata flows to both ZenML and Modal Volume

#### Storage Operations:

- Model saved locally and to Modal Volume
- Fairness reports saved to Modal Volume
- Risk register updated in Excel and exported as Markdown
- All key metrics and metadata logged to ZenML

### Deployment Pipeline

The Deployment Pipeline handles the model deployment process with human oversight, documentation generation, and post-market monitoring setup.

#### Step-by-Step Process:

1. **Approval Process (Article 14 Compliance):**

   - Presents comprehensive performance and fairness metrics to reviewer
   - Requires explicit human approval for deployment
   - Supports both interactive and automated approval flows
   - Records approval decision, rationale, and approver identity
   - Creates and stores detailed approval record for auditability
   - Enforces hard-stop if approval is denied

2. **Modal Deployment (Articles 10, 17, 18 Compliance):**

   - Loads deployment code dynamically from application path
   - Launches Modal deployment process for serverless hosting
   - Creates comprehensive deployment record with model checksum
   - Generates model card with performance metrics and fairness considerations
   - Saves compliance artifacts to Modal Volume
   - Returns deployment information including API endpoints

3. **Post-Market Monitoring (Article 17 Compliance):**

   - Creates comprehensive monitoring plan based on model metrics
   - Defines monitoring frequency for data drift, performance, and fairness
   - Sets alert thresholds derived from baseline performance
   - Documents response procedures for different severity levels
   - Assigns responsibilities for monitoring and escalation
   - Saves monitoring plan to compliance directory

4. **Annex IV Documentation (Comprehensive Compliance):**
   - Collects metadata from ZenML context and pipeline runs
   - Loads manual inputs from YAML configuration files
   - Renders Jinja template with comprehensive compliance information
   - Identifies missing fields requiring manual input
   - Saves documentation to compliance reports directory and Modal Volume
   - Supports optional PDF generation

#### Data Flow:

- Artifacts from previous pipelines → Deployment steps
- Approval decision → Modal deployment
- Deployment info → Post-market monitoring
- All metadata → Annex IV documentation

#### Storage Operations:

- Approval records saved to compliance/approval_records/
- Monitoring plan saved to compliance/monitoring/
- Annex IV documentation saved to compliance/reports/
- All artifacts also saved to Modal Volume for persistence

### Modal Deployment Implementation

The Modal Deployment implements the serverless API with comprehensive monitoring and incident reporting capabilities.

#### Step-by-Step Process:

1. **Application Setup:**

   - Creates Modal app with necessary dependencies
   - Sets up volume mounts for model and pipeline access
   - Defines functions for model loading, prediction, monitoring, and incident reporting

2. **API Implementation:**

   - Creates FastAPI application with documented endpoints
   - Implements health check, prediction, monitoring, and incident reporting endpoints
   - Ensures proper error handling and logging throughout

3. **Prediction Flow:**

   - Loads model and preprocessing pipeline from Modal Volume
   - Processes input data using the saved preprocessing pipeline
   - Makes prediction with the trained model
   - Returns prediction with risk assessment and model version

4. **Monitoring Implementation:**

   - Implements data drift detection function
   - Sets up incident reporting for detected issues
   - Provides endpoint for manual triggering of monitoring

5. **Deployment Process:**
   - Deploys FastAPI app to Modal
   - Records deployment details including endpoints
   - Creates comprehensive model card with performance metrics
   - Returns deployment record and model card to calling pipeline

#### Data Flow:

- Input requests → FastAPI → Model/Pipeline → Prediction response
- Monitoring triggers → Drift detection → Incident reporting if needed
- Deployment parameters → Modal app → Deployment record

#### Storage Operations:

- Loads model and pipeline from Modal Volume
- Logs incidents to temporary storage and optional webhook
- Returns deployment metadata to pipeline for persistent storage
