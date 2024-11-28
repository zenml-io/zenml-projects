# Train and Deploy ML Project

This README provides step-by-step instructions for running the training and deployment pipeline using ZenML.

## Prerequisites

- Git installed
- Python environment set up
- ZenML installed
- Access to the ZenML project repository

## Project Setup

1. Clone the repository and checkout the feature branch:
```bash
git clone git@github.com:zenml-io/zenml-projects.git
git checkout feature/update-train-deploy
```

2. Navigate to the project directory:
```bash
cd train_and_deploy
```

3. Initialize ZenML in the project:
```bash
zenml init
```

## Running the Pipeline

### Training

You have two options for running the training pipeline:

#### Option 1: Automatic via CI
Make any change to the code and push it. This will automatically trigger the CI pipeline that launches training in SkyPilot.

#### Option 2: Manual Execution
1. First, set up your stack. You can choose between:
   - Local stack (uses local orchestrator):
     ```bash
     zenml stack set LocalGitGuardian
     ```
   - Remote stack (uses SkyPilot orchestrator):
     ```bash
     zenml stack set RemoteGitGuardian
     ```

2. Run the training pipeline:
```bash
python run --training
```

### Model Deployment

1. After training completes, deploy the model:
```bash
python run --deployment
```

Note: At this stage, the deployment is done to the model set as "staging" (configured in `target_env`), and the model is deployed locally using BentoML.

2. Test the deployed model:
```bash
python run --inference
```

### Production Deployment

If the staging model performs well and you want to proceed with production deployment:

1. Deploy to Kubernetes:
```bash
python run --production
```
This pipeline will:
- Build a Docker image from the BentoML service
- Deploy it to Kubernetes

## Additional Resources

- [ZenML Projects Tenant Dashboard](https://cloud.zenml.io/organizations/fc992c14-d960-4db7-812e-8f070c99c6f0/tenants/12ec0fd2-ed02-4479-8ff9-ecbfbaae3285)
- [Example GitHub Actions Pipeline](https://github.com/zenml-io/zenml-projects/actions/runs/12075854945/job/33676323427)

## Pipeline Flow Overview

1. Training → Creates and trains the model
2. Deployment → Deploys model to staging environment (local BentoML)
3. Inference → Tests the deployed model
4. Production → Deploys to production Kubernetes environment

## Notes

- The deployment configurations are controlled by the `target_env` setting in the configs
- Make sure you have the necessary permissions and access rights before running the pipelines
- Monitor the CI/CD pipeline in GitHub Actions when using automatic deployment