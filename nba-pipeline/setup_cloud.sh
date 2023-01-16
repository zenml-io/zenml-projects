#!/usr/bin/env bash

set -Eeo pipefail

# These settings are hard-coded at the moment
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 715803424590.dkr.ecr.us-east-1.amazonaws.com
aws eks --region us-east-1 update-kubeconfig --name zenhacks-cluster --alias zenml-eks

zenml secrets-manager register aws_secrets_manager --flavor=aws --region_name=eu-central-1
zenml experiment-tracker register aws_mlflow_tracker  --flavor=mlflow --tracking_insecure_tls=true --tracking_uri="https://ac8e6c63af207436194ab675ee71d85a-1399000870.us-east-1.elb.amazonaws.com/mlflow" --tracking_username="{{mlflow_secret.tracking_username}}" --tracking_password="{{mlflow_secret.tracking_password}}" 
zenml orchestrator register multi_tenant_kubeflow \
  --flavor=kubeflow \
  --kubernetes_context=kubeflowmultitenant \
  --kubeflow_hostname=https://www.kubeflowshowcase.zenml.io/pipeline

zenml artifact-store register s3_store -f s3 --path=s3://zenml-projects

zenml container-registry register ecr_registry --flavor=aws --uri=715803424590.dkr.ecr.us-east-1.amazonaws.com 

# For GKE clusters, the host is the GKE cluster IP address.
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
# For EKS clusters, the host is the EKS cluster IP hostname.
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
export INGRESS_URL="http://${INGRESS_HOST}:${INGRESS_PORT}"

zenml model-deployer register kserve_s3 --flavor=kserve --kubernetes_context=kubeflowmultitenant  --kubernetes_namespace=zenml-workloads --base_url=$INGRESS_URL --secret=kservesecret 

zenml stack register kubeflow_gitflow_stack \
    -a s3_store \
    -c ecr_registry \
    -o multi_tenant_kubeflow \
    -x aws_secrets_manager \
    -d kserve_s3 \
    -e aws_mlflow_tracker || \
  msg "${WARNING}Reusing preexisting stack ${NOFORMAT}kubeflow_gitflow_stack"

zenml stack set kubeflow_gitflow_stack
zenml stack share kubeflow_gitflow_stack

zenml secrets-manager secret register -s kserve_s3 kservesecret --credentials="@~/.aws/credentials" 

echo "In the following prompt, please set the `tracking_username` key with value of your MLflow username and `tracking_password` key with value of your MLflow password. "
zenml secrets-manager secret register mlflow_secret -i
