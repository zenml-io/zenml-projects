#!/usr/bin/env bash

set -Eeo pipefail

# These settings are hard-coded at the moment
export GOOGLE_APPLICATION_CREDENTIALS=~/.ssh/google-demos.json
gcloud auth configure-docker --project zenml-demos 

zenml data-validator register deepchecks_data_validator --flavor=deepchecks
zenml experiment-tracker register gcp_mlflow_tracker  --flavor=mlflow --tracking_insecure_tls=true --tracking_uri="http://35.246.148.181/mlflow/" --tracking_username="{{mlflow_secret.tracking_username}}" --tracking_password="{{mlflow_secret.tracking_password}}" 
zenml orchestrator register vertex_ai_orchestrator \
  --flavor=vertex \
  --project=zenml-demos \
  --location=europe-west3 \
  --workload_service_account=ing-zenmlsa-ing@zenml-demos.iam.gserviceaccount.com \
  --synchronous=true

zenml artifact-store register gcp_store -f gcp --path=gs://ing-store

zenml image-builder register local_image_builder -f local
zenml container-registry register gcp_registry --flavor=gcp --uri=eu.gcr.io/zenml-demos 

zenml stack register gcp_gitflow_stack \
    -a gcp_store \
    -c gcp_registry \
    -o vertex_ai_orchestrator \
    -dv deepchecks_data_validator \
    -e gcp_mlflow_tracker \
    -i local_image_builder || \
  msg "${WARNING}Reusing preexisting stack ${NOFORMAT}gcp_gitflow_stack"

zenml stack set gcp_gitflow_stack
zenml stack share gcp_gitflow_stack

echo "In the following prompt, please set the `tracking_username` key with value of your MLflow username and `tracking_password` key with value of your MLflow password. "
zenml secret create mlflow_secret -i  # set tracking_username and tracking_password