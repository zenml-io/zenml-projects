# 🌸 FloraCast

A production-ready MLOps pipeline for time series forecasting using ZenML and Darts, designed for DFG's forecasting needs.

## 🚀 Product Overview

FloraCast demonstrates how to build end-to-end MLOps workflows for time series forecasting. Built with ZenML's robust framework, it showcases enterprise-grade machine learning pipelines that can be deployed in both development and production environments.

### Key Features

- **End-to-End Forecasting Pipeline**: From data ingestion to model deployment
- **Darts Integration**: Support for advanced forecasting models like TFT (Temporal Fusion Transformer)
- **Custom Materializers**: Production-ready artifact handling with visualizations
- **Model Versioning**: Track and compare different model versions
- **Flexible Configuration**: YAML-based configuration for different environments
- **Cloud Ready**: Built with EKS/GKE/AKS deployment in mind

## 💡 How It Works

FloraCast consists of two main pipelines:

### 1. Training Pipeline

The training pipeline handles the complete ML workflow:

1. **Data Ingestion** - Loads ecommerce sales data (synthetic by default)
2. **Preprocessing** - Converts to Darts TimeSeries with train/validation split  
3. **Model Training** - Trains TFT model with configurable parameters
4. **Evaluation** - Computes SMAPE metrics on validation set
5. **Model Registration** - Registers model artifacts for tracking

```bash
python run.py --config configs/training.yaml --pipeline train
```

### 2. Batch Inference Pipeline

The inference pipeline generates predictions using trained models:

1. **Data Ingestion** - Loads new data for predictions
2. **Preprocessing** - Applies same transformations as training
3. **Model Loading** - Loads most recent trained model
4. **Batch Inference** - Generates forecasts and saves to CSV

```bash
python run.py --config configs/inference.yaml --pipeline inference
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- ZenML account (optional for cloud features)
- Virtual environment (recommended)

### Setup

1. **Clone the repository** (if part of zenml-projects):
```bash
cd zenml-projects/floracast
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment** (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

## ⚡ Quick Start

### Local Development

1. **Run training pipeline**:
```bash
python run.py --config configs/training.yaml --pipeline train
```

2. **Run inference pipeline**:
```bash  
python run.py --config configs/inference.yaml --pipeline inference
```

3. **View results**:
- Check `outputs/forecast_inference.csv` for predictions
- Use ZenML dashboard to view artifacts and metrics

## ⚙️ Configuration Files

FloraCast uses semantically named configuration files for different deployment scenarios:

### Available Configurations

- **`configs/training.yaml`** - Local development and training pipeline configuration
- **`configs/inference.yaml`** - Batch inference pipeline configuration for production models  

### Customization Options

Edit the appropriate config file to customize:

- **Model parameters**: TFT hyperparameters, training epochs
- **Data settings**: Date columns, frequency, validation split
- **Evaluation**: Forecasting horizon, metrics
- **Output**: File paths and formats

Example configuration:
```yaml
model:
  name: "TFTModel"
  params:
    input_chunk_length: 30
    output_chunk_length: 7
    hidden_size: 32
    n_epochs: 10
    add_relative_index: true

evaluation:
  horizon: 7
  metric: "smape"
```

## 🔧 Architecture

### Directory Structure

```
floracast/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── training.yaml       # Training pipeline config
│   ├── inference.yaml      # Inference pipeline config  
├── data/
│   └── ecommerce_daily.csv # Generated sample data
├── outputs/                # Inference results
├── pipelines/
│   ├── train_forecast_pipeline.py
│   └── batch_inference_pipeline.py
├── steps/
│   ├── ingest.py          # Data loading
│   ├── preprocess.py      # Time series preprocessing  
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
│   ├── promote.py         # Model registration
│   ├── batch_infer.py     # Batch predictions
│   └── load_model.py      # Model loading utilities
├── materializers/
│   └── darts_materializer.py  # Custom TimeSeries materializer
├── utils/
│   └── metrics.py         # Forecasting metrics
└── run.py                 # Main entry point
```

### Key Components

- **Custom Materializers**: Proper serialization for Darts TimeSeries with visualizations
- **Flexible Models**: TFT primary, ExponentialSmoothing fallback
- **Comprehensive Logging**: Detailed pipeline execution tracking
- **Artifact Naming**: Clear, descriptive names for all pipeline outputs

## 📊 Features in Detail

### Time Series Forecasting

- **Multiple Models**: TFT (Temporal Fusion Transformer) and ExponentialSmoothing
- **Automated Feature Engineering**: Relative index encoding for TFT
- **Flexible Horizons**: Configurable forecasting windows
- **Performance Metrics**: SMAPE evaluation with fallback handling

### MLOps Best Practices

- **Artifact Versioning**: Every model run is tracked and versioned
- **Pipeline Caching**: Intelligent caching to avoid redundant computation  
- **Configuration Management**: Environment-specific YAML configs
- **Error Handling**: Graceful fallbacks and comprehensive logging

### Production Ready

- **Custom Materializers**: Production-grade artifact serialization
- **Visualization Support**: Built-in time series plots in ZenML dashboard
- **Metadata Tracking**: Rich metadata for all artifacts and models
- **Scalable Architecture**: Ready for cloud deployment

## 🚀 Production Deployment

### ZenML Azure Stack Setup

To run FloraCast on Azure with ZenML, set up a ZenML stack backed by Azure services:

- **Artifact Store**: Azure Blob Storage
- **Container Registry**: Azure Container Registry (ACR)
- **Orchestrator**: Kubernetes Orchestrator targeting AKS
- **Optional**: AzureML Step Operator for managed training; Azure Key Vault for secrets

Quick start (CLI):

```bash
# Artifact Store (Azure Blob)
zenml artifact-store register azure_store --flavor=azure \
  --account_name=<AZURE_STORAGE_ACCOUNT> \
  --container=<AZURE_STORAGE_CONTAINER>

# Container Registry (ACR)
zenml container-registry register azure_acr --flavor=azure \
  --uri=<ACR_LOGIN_SERVER>

# Orchestrator (AKS via Kubernetes)
zenml orchestrator register aks_k8s --flavor=kubernetes \
  --kubernetes_context=<AKS_KUBE_CONTEXT> \
  --namespace=<NAMESPACE>

# (Optional) AzureML Step Operator
zenml step-operator register azureml_ops --flavor=azureml \
  --subscription_id=<SUBSCRIPTION_ID> \
  --resource_group=<RESOURCE_GROUP> \
  --workspace_name=<AML_WORKSPACE>

# Compose the stack
zenml stack register azure_aks_stack \
  -a azure_store -c azure_acr -o aks_k8s --set
```

Read more:

- **Set up an MLOps stack on Azure**: [ZenML Azure guide](https://docs.zenml.io/stacks/popular-stacks/azure-guide)
- **Kubernetes Orchestrator (AKS)**: [Docs](https://docs.zenml.io/stacks/stack-components/orchestrators/kubernetes)
- **Azure Blob Artifact Store**: [Docs](https://docs.zenml.io/stacks/stack-components/artifact-stores/azuree)
- **Azure Container Registry**: [Docs](https://docs.zenml.io/stacks/stack-components/container-registries/azure)
- **AzureML Step Operator**: [Docs](https://docs.zenml.io/stacks/stack-components/step-operators/azureml)
- **Terraform stack recipe for Azure**: [Hashicorp Registry](https://registry.terraform.io/modules/zenml-io/zenml-stack/azure/latest)

### Azure Kubernetes Service (AKS)

The project includes configuration for AKS deployment:

**Prerequisites**:
   - AKS cluster configured
   - Azure Container Registry (ACR)  
   - Azure Blob Storage for artifacts

### Scaling Considerations

- **Model Parameters**: AKS config uses larger model sizes and longer training
- **Resource Allocation**: Configure memory/CPU limits for forecasting workloads
- **Storage**: Use Azure Blob for artifact persistence across pipeline runs

## 🔬 Advanced Usage

### Custom Data Sources

Replace the default ecommerce data:

1. **Update configuration**:
```yaml
data:
  source: "csv"
  path: "path/to/your/data.csv"
  datetime_col: "timestamp"
  target_col: "sales"
```

2. **Ensure data format**:
   - DateTime index column
   - Numeric target variable  
   - Daily frequency (or update `freq` parameter)

### Model Experimentation

Try different forecasting models by updating the config:

```yaml
model:
  name: "ExponentialSmoothing"  # Fallback model
  params:
    seasonal_periods: 7
```

### Custom Metrics

Extend `utils/metrics.py` to add additional forecasting metrics:

```python
def mase(actual: TimeSeries, predicted: TimeSeries) -> float:
    # Mean Absolute Scaled Error implementation
    pass
```

## 🤝 Contributing

FloraCast follows ZenML best practices and is designed to be extended:

1. **Add New Models**: Extend `steps/train.py` with additional Darts models
2. **Custom Materializers**: Create materializers for new data types  
3. **Additional Metrics**: Expand evaluation capabilities
4. **New Data Sources**: Add support for different input formats

## 📝 Next Steps

After running FloraCast successfully:

1. **Explore ZenML Dashboard**: View pipeline runs, artifacts, and metrics
2. **Experiment with Models**: Try different TFT configurations
3. **Add Real Data**: Replace synthetic data with your forecasting use case
4. **Deploy to Production**: Use AKS configuration for scale
5. **Model Registry**: Implement proper model promotion workflows

## 🆘 Troubleshooting

### Common Issues

**TFT Training Fails**:
- Check `add_relative_index: true` in configuration  
- Verify sufficient data length (>30 points for input_chunk_length=30)

**Materializer Errors**:
- Ensure datetime columns are properly formatted
- Check that TimeSeries can be created from your data

**Memory Issues**:
- Reduce `batch_size` or `hidden_size` in model parameters
- Use ExponentialSmoothing for lighter resource usage

## 📚 Resources

- [ZenML Documentation](https://docs.zenml.io/)
- [Darts Documentation](https://unit8co.github.io/darts/)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)

---

Built with ❤️ using [ZenML](https://zenml.io) - *The MLOps Framework for Production AI*