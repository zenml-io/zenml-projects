# 📜 ZenML Stack Show Case

This project aims to demonstrate the power of stacks. The code in this 
project assumes that you ave quite a few stacks registered already:

* default
* local-sagemaker-step-operator-stack
  * `default` Orchestrator
  * `s3` Artifact Store
  * `local` Image Builder
  * `aws` Container Registry
  * `Sagemaker` Step Operator
* sagemaker-airflow-stack
  * `Airflow` Orchestrator
  * `s3` Artifact Store
  * `local` Image Builder
  * `aws` Container Registry
  * `Sagemaker` Step Operator
* sagemaker-stack
  * `Sagemaker` Orchestrator
  * `s3` Artifact Store
  * `local` Image Builder
  * `aws` Container Registry
  * `Sagemaker` Step Operator