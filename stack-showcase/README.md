
---
title: ZenML Breast Cancer Classifier
emoji: 🦀
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 📜 ZenML Stack Show Case

This project aims to demonstrate the power of stacks. The code in this 
project assumes that you have quite a few stacks registered already. 

## default
  * `default` Orchestrator
  * `default` Artifact Store

```commandline
zenml stack set default
python run.py --training-pipeline
```

## local-sagemaker-step-operator-stack
  * `default` Orchestrator
  * `s3` Artifact Store
  * `local` Image Builder
  * `aws` Container Registry
  * `Sagemaker` Step Operator

```commandline
zenml stack set local-sagemaker-step-operator-stack
zenml integration install aws -y
python run.py --training-pipeline
```

## sagemaker-airflow-stack
  * `Airflow` Orchestrator
  * `s3` Artifact Store
  * `local` Image Builder
  * `aws` Container Registry
  * `Sagemaker` Step Operator

```commandline
zenml stack set sagemaker-airflow-stack
zenml integration install airflow -y
pip install apache-airflow-providers-docker apache-airflow~=2.5.0
zenml stack up
python run.py --training-pipeline
```

## sagemaker-stack
  * `Sagemaker` Orchestrator
  * `s3` Artifact Store
  * `local` Image Builder
  * `aws` Container Registry
  * `Sagemaker` Step Operator

```commandline
zenml stack set sagemaker-stack
python run.py --training-pipeline
```
