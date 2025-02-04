# 🚀 Deploying ML Models with ZenML on Vertex AI


Welcome to your ZenML project for deploying ML models using Google Cloud's Vertex AI! This project provides a hands-on experience with MLOps pipelines using ZenML and Vertex AI. It contains a collection of ZenML steps, pipelines, and other artifacts to help you efficiently deploy your machine learning models.

Using these pipelines, you can run data preparation, model training, registration, and deployment with a single command while using YAML files for [configuration](https://docs.zenml.io/user-guide/production-guide/configure-pipeline). ZenML takes care of tracking your metadata and [containerizing your pipelines](https://docs.zenml.io/how-to/customize-docker-builds).


## 🏃 How to run

In this project, we will train and deploy a classification model to [Vertex AI](https://cloud.google.com/vertex-ai). Before running any pipelines, set up your environment as follows, we need to set up our environment as follows:

```bash
# Set up a Python virtual environment, if you haven't already
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

We will need to set up access to Google Cloud and Vertex AI. You can follow the instructions in the [ZenML documentation](https://docs.zenml.io/how-to/auth-management/gcp-service-connector)
to register a service connector and set up your Google Cloud credentials.

Once you have set up your Google Cloud credentials, we can create a stack and run the deployment pipeline:

```bash
# Register the artifact store
zenml artifact-store register gs_store -f gcp --path=gs://zenml-vertex-test
zenml artifact-store connect gs_store --connector gcp

# Register the model registry
zenml model-registry register vertex_registry --flavor=vertex --location=europe-west1 
zenml model-registry connect vertex_registry --connector gcp

# Register Model Deployer
zenml model-deployer register vertex_deployer --flavor=vertex --location=europe-west1
zenml model-deployer connect vertex_deployer --connector gcp

# Register the stack
zenml stack register vertex_stack --orchestrator default --artifact-store gs_store --model-registry vertex_registry --model-deployer vertex_deployer
```

Now that we have set up our stack, we can run the training pipeline, which will train and register the model into the Vertex AI model registry and Deploys it into Vertex AI endpoint.

```bash
python run.py --training-pipeline
```

Once the pipeline has completed, you can check the status of the model in the Vertex AI model registry and the deployed model in the Vertex AI endpoint.

```bash
# List models in the model registry
zenml model-registry models list

# List deployed models
zenml model-deployer models list
```

You can also run the deployment pipeline separately:

```bash
python run.py --inference-pipeline
```


## 📜 Project Structure

The project loosely follows [the recommended ZenML project structure](https://docs.zenml.io/how-to/setting-up-a-project-repository/best-practices):

```
.
├── configs                                     # Pipeline configuration files
│   ├── training.yaml                           # Configuration for training pipeline
│   ├── inference.yaml                          # Configuration for inference pipeline
├── pipelines                                   # `zenml.pipeline` implementations
│   ├── training.py                             # Training pipeline
│   ├── inference.py                            # Inference pipeline
├── steps                                       # `zenml.step` implementations
│   ├── model_trainer.py                        # Model training step
│   ├── model_register.py                       # Model registration step
│   ├── model_promoter.py                       # Model promotion step
│   ├── model_deployer.py                       # Model deployment step to Vertex AI
├── README.md                                   # This file
├── requirements.txt                            # Extra Python dependencies
└── run.py                                      # CLI tool to run pipelines with ZenML                                    # CLI tool to run pipelines on ZenML Stack
```