#!/bin/bash

# Setup script for FloraCast local development stack

set -e

echo "🌸 Setting up FloraCast local development stack..."

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Set default values
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"http://localhost:5000"}

echo "Using MLflow tracking URI: $MLFLOW_TRACKING_URI"

# Initialize ZenML repository (skip if already exists)
echo "Initializing ZenML repository..."
zenml init || echo "ZenML repository already exists"

# Register MLflow experiment tracker
echo "Registering MLflow experiment tracker..."
zenml experiment-tracker register floracast_mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri="$MLFLOW_TRACKING_URI" || echo "MLflow tracker already exists"

# Register local orchestrator
echo "Registering local orchestrator..."
zenml orchestrator register floracast_local_orchestrator \
    --flavor=local || echo "Local orchestrator already exists"

# Register local artifact store
echo "Registering local artifact store..."
zenml artifact-store register floracast_local_artifact_store \
    --flavor=local \
    --path=./.zenml_artifacts || echo "Local artifact store already exists"

# Create local stack
echo "Creating local stack..."
zenml stack register floracast_local_stack \
    -o floracast_local_orchestrator \
    -a floracast_local_artifact_store \
    -e floracast_mlflow_tracker || echo "Local stack already exists"

# Activate the stack
echo "Activating local stack..."
zenml stack set floracast_local_stack

# Verify stack
echo "Verifying stack configuration..."
zenml stack describe

echo "✅ Local stack setup completed successfully!"
echo "You can now run: python run.py --config configs/local.yaml --pipeline train"