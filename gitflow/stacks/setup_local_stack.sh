#!/usr/bin/env bash

set -Eeo pipefail

zenml data-validator register deepchecks_data_validator --flavor=deepchecks
zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow
zenml model-deployer register local_mlflow_deployer  --flavor=mlflow
zenml stack register local_gitflow_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker \
    -d local_mlflow_deployer \
    -dv deepchecks_data_validator
zenml stack set local_gitflow_stack