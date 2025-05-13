# Pipeline Architecture Analysis

## 1. DETAILED PIPELINE EXPLANATIONS

### Feature Engineering Pipeline

The Feature Engineering Pipeline forms the foundation of your credit scoring system, focused on data preparation while ensuring EU AI Act compliance.

#### Step-by-Step Process:

1. **Data Loading (Article 10 Compliance):**

   - Sources data from HuggingFace (specifically from `HF_DATASET_NAME` constant)
   - Calculates SHA-256 hash for data provenance and traceability
   - Creates WhyLogs profile for data quality documentation and returns it directly as a ZenML artifact
   - Identifies potential sensitive attributes automatically
   - Logs comprehensive metadata to ZenML for lineage tracking
   - Samples data if requested (using stratified sampling to maintain class distribution)

2. **Data Splitting:**

   - Implements a train-test split using scikit-learn
   - Preserves dataset schema and applies stratification
   - Returns clearly named raw training and test datasets for further processing

3. **Data Preprocessing (Article 10 Compliance):**

   - Handles missing values by dropping NA values if requested
   - Removes specified columns (automatic removal of sensitive data like wallet addresses)
   - Standardizes numerical features using StandardScaler if normalization is requested
   - Creates and fits a preprocessing pipeline using scikit-learn's ColumnTransformer
   - Saves preprocessing pipeline to Modal Volume using centralized constants
   - Logs checksums and transformation details to ensure data providence

4. **Compliance Metadata Generation (Articles 10, 12, 15):**
   - Receives WhyLogs data profile directly from data_loader step
   - Documents feature specifications, types, and statistics
   - Identifies and flags sensitive attributes for fairness checks
   - Tracks data quality metrics before and after preprocessing
   - Creates comprehensive compliance record for documentation
   - Stores metadata in ZenML for traceability

#### Data Flow:

- Raw data from HuggingFace → Data loader → Data splitter → Preprocessor → Final training/test splits
- Data profile and preprocessing metadata flow through ZenML artifacts
- Preprocessing pipeline saved to Modal Volume using centralized configuration

#### Storage Operations:

- Preprocessing pipeline saved to Modal Volume at standard path from `constants.py`
- WhyLogs profile passed directly as ZenML artifact
- Compliance metadata logged to ZenML via log_metadata()
- Pipeline returns structured data artifacts for the training pipeline

### Training Pipeline

The Training Pipeline implements model training with comprehensive fairness evaluation and risk assessment mechanisms to ensure EU AI Act compliance.

#### Step-by-Step Process:

1. **Model Training (Article 11 Compliance):**

   - Supports training with either pipeline-supplied data or automatic retrieval from ZenML artifacts
   - Trains a GradientBoostingClassifier with configurable hyperparameters
   - Handles preprocessed feature names with transformation suffixes
   - Records training parameters, start time, and duration
   - Returns model directly as a ZenML artifact
   - Saves model checksum to Modal Volume using constants from centralized configuration
   - Creates a model card with purpose, limitations, and performance metrics
   - Logs model metadata for integrity verification

2. **Model Evaluation (Articles 9, 15 Compliance):**

   - Receives model directly as a ZenML artifact
   - Calculates performance metrics (accuracy, AUC)
   - Performs fairness analysis using Fairlearn's MetricFrame
   - Checks selection rates and accuracy across protected attributes
   - Flags bias if selection rate disparity exceeds threshold (0.2)
   - Captures comprehensive fairness metrics per demographic group
   - Returns evaluation results as ZenML artifacts for subsequent steps

3. **Risk Assessment (Article 9 Compliance):**
   - Processes evaluation results to calculate risk scores
   - Updates risk register in Excel format with quantified risk metrics
   - Creates a Markdown snapshot of the risk register
   - Categorizes risk level and suggests mitigations
   - Returns risk assessment as ZenML artifact for deployment pipeline

#### Data Flow:

- Training/test datasets → Model training → Model object → Evaluation
- Evaluation results → Risk assessment
- All artifacts flow through ZenML with consistent naming

#### Storage Operations:

- Model artifact stored in ZenML and to Modal Volume
- Model checksum saved to Modal Volume
- Risk register updated in Excel and exported as Markdown to Modal Volume
- All key metrics and metadata logged to ZenML

### Deployment Pipeline

The Deployment Pipeline handles the model deployment process with human oversight, documentation generation, and post-market monitoring setup.

#### Step-by-Step Process:

1. **Approval Process (Article 14 Compliance):**

   - Automatically fetches artifacts from ZenML if not provided
   - Presents comprehensive performance and fairness metrics to reviewer
   - Requires explicit human approval for deployment
   - Supports both interactive and automated approval flows
   - Records approval decision, rationale, and approver identity
   - Creates and returns approval record as ZenML artifact
   - Enforces hard-stop if approval is denied

2. **Modal Deployment (Articles 10, 17, 18 Compliance):**

   - Generates model checksum from model object
   - Creates comprehensive deployment record with model checksum
   - Loads deployment code from application directory
   - Launches Modal deployment process for serverless hosting
   - Generates model card with performance metrics and fairness considerations
   - Returns deployment information including API endpoints

3. **Post-Market Monitoring (Article 17 Compliance):**

   - Creates comprehensive monitoring plan based on model metrics
   - Defines monitoring frequency for data drift, performance, and fairness
   - Sets alert thresholds derived from baseline performance
   - Documents response procedures for different severity levels
   - Assigns responsibilities for monitoring and escalation
   - Returns monitoring plan as ZenML artifact

4. **Annex IV Documentation (Comprehensive Compliance):**
   - Collects metadata from ZenML context and pipeline runs
   - Loads manual inputs from YAML configuration files
   - Renders Jinja template with comprehensive compliance information
   - Identifies missing fields requiring manual input
   - Saves documentation to compliance reports directory
   - Supports optional PDF generation

#### Data Flow:

- All artifacts flow through ZenML with consistent naming from `constants.py`
- Approval decision → Modal deployment
- Deployment info → Post-market monitoring
- All metadata → Annex IV documentation

#### Storage Operations:

- ZenML for artifact storage and metadata tracking
- Modal Volume for deployment and compliance artifacts with standardized paths
- Annex IV documentation saved locally for manual review

### Modal Deployment Implementation

The Modal Deployment implements the serverless API with comprehensive monitoring and incident reporting capabilities.

#### Step-by-Step Process:

1. **Application Setup:**

   - Creates Modal app with necessary dependencies
   - Sets up volume mounts for model and pipeline access
   - Uses centralized configuration from `constants.py` for consistency
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

- Loads model and pipeline from standardized Modal Volume paths
- Logs incidents to temporary storage and optional webhook
- Returns deployment metadata to pipeline for persistent storage

## 2. Key Architecture Improvements

### Centralized Configuration

- All constants defined in `constants.py` for consistent reference across pipelines
- Standard paths for Modal Volume storage with `VOLUME_METADATA_KEYS` dictionary
- Consistent artifact naming for ZenML tracking

### Streamlined Data Flow

- Direct ZenML artifact passing between pipeline steps
- Data profile passed directly from data_loader to compliance metadata generation
- Model object passed directly between training and evaluation steps
- Automatic fallback to ZenML artifact retrieval when artifacts not provided

### Efficient Storage Strategy

- Model stored directly as ZenML artifact without redundant local saving
- Modal Volume used only for deployment artifacts
- Preprocessing pipeline saved to Modal Volume for prediction
- Eliminated redundant local saving of intermediate artifacts

### Improved Compliance Documentation

- WhyLogs profile created and passed directly as ZenML artifact
- Comprehensive metadata logging for all pipeline steps
- Centralized paths for compliance documentation
- Standardized model card generation
