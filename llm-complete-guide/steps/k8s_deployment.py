#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import re
from pathlib import Path
from typing import Dict, Optional, cast

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from zenml import get_step_context, step
from zenml.client import Client
from zenml.integrations.bentoml.services.bentoml_local_deployment import (
    BentoMLLocalDeploymentConfig,
    BentoMLLocalDeploymentService,
)
from zenml.logger import get_logger
from zenml.orchestrators.utils import get_config_environment_vars

logger = get_logger(__name__)

def apply_kubernetes_configuration(k8s_configs: list) -> None:
    """Apply Kubernetes configurations using the K8s Python client.
    
    Args:
        k8s_configs: List of Kubernetes configuration dictionaries
    """
    # Load Kubernetes configuration
    try:
        config.load_kube_config()
    except:
        config.load_incluster_config()  # For in-cluster deployment
        
    # Initialize API clients
    k8s_apps_v1 = client.AppsV1Api()
    k8s_core_v1 = client.CoreV1Api()
    
    for k8s_config in k8s_configs:
        kind = k8s_config["kind"]
        name = k8s_config["metadata"]["name"]
        namespace = k8s_config["metadata"].get("namespace", "default")
        
        try:
            if kind == "Deployment":
                # Check if deployment exists
                try:
                    k8s_apps_v1.read_namespaced_deployment(name, namespace)
                    # Update existing deployment
                    k8s_apps_v1.patch_namespaced_deployment(
                        name=name,
                        namespace=namespace,
                        body=k8s_config
                    )
                    logger.info(f"Updated existing deployment: {name}")
                except ApiException as e:
                    if e.status == 404:
                        # Create new deployment
                        k8s_apps_v1.create_namespaced_deployment(
                            namespace=namespace,
                            body=k8s_config
                        )
                        logger.info(f"Created new deployment: {name}")
                    else:
                        raise e
                        
            elif kind == "Service":
                # Check if service exists
                try:
                    k8s_core_v1.read_namespaced_service(name, namespace)
                    # Update existing service
                    k8s_core_v1.patch_namespaced_service(
                        name=name,
                        namespace=namespace,
                        body=k8s_config
                    )
                    logger.info(f"Updated existing service: {name}")
                except ApiException as e:
                    if e.status == 404:
                        # Create new service
                        k8s_core_v1.create_namespaced_service(
                            namespace=namespace,
                            body=k8s_config
                        )
                        logger.info(f"Created new service: {name}")
                    else:
                        raise e
                        
        except ApiException as e:
            logger.error(f"Error applying {kind} {name}: {e}")
            raise e

@step(enable_cache=False)
def k8s_deployment(
    docker_image_tag: str,
    namespace: str = "default"
) -> Dict:
    # Get the raw model name
    raw_model_name = get_step_context().model.name
    # Sanitize the model name
    model_name = sanitize_name(raw_model_name)
    
    # Get environment variables
    environment_vars = get_config_environment_vars()
    
    # Get current deployment
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    services = model_deployer.find_model_server(
        model_name=model_name,
        model_version="production",
    )

    # Read the K8s template
    template_path = Path(__file__).parent / "k8s_template.yaml"
    with open(template_path, "r") as f:
        k8s_configs = list(yaml.safe_load_all(f))
    
    # Update configurations with sanitized names
    for config in k8s_configs:
        # Add namespace
        config["metadata"]["namespace"] = namespace
        
        # Update metadata labels and name
        config["metadata"]["labels"]["app"] = model_name
        config["metadata"]["name"] = model_name
        
        if config["kind"] == "Service":
            # Update service selector
            config["spec"]["selector"]["app"] = model_name

            # Update metadata annotations with SSL certificate ARN
            config["metadata"]["annotations"] = {
                "service.beta.kubernetes.io/aws-load-balancer-ssl-cert": "arn:aws:acm:eu-central-1:339712793861:certificate/0426ace8-5fa3-40dd-bd81-b0fb1064bd85",
                "service.beta.kubernetes.io/aws-load-balancer-backend-protocol": "http",
                "service.beta.kubernetes.io/aws-load-balancer-ssl-ports": "443",
                "service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout": "3600"
            }
 
            # Update ports
            config["spec"]["ports"] = [
                {
                    "name": "https",
                    "port": 443,
                    "targetPort": 3000
                }
            ]
            
        elif config["kind"] == "Deployment":
            # Update deployment selector and template
            config["spec"]["selector"]["matchLabels"]["app"] = model_name
            config["spec"]["template"]["metadata"]["labels"]["app"] = model_name
            
            # Update the container image and name
            containers = config["spec"]["template"]["spec"]["containers"]
            for container in containers:
                container["name"] = model_name
                container["image"] = docker_image_tag
        
                # Add environment variables to the container
                env_vars = []
                for key, value in environment_vars.items():
                    env_vars.append({"name": key, "value": value})
                container["env"] = env_vars
    
    # Apply the configurations
    try:
        apply_kubernetes_configuration(k8s_configs)
        deployment_status = "success"
        logger.info(f"Successfully deployed model {model_name} with image: {docker_image_tag}")
    except Exception as e:
        deployment_status = "failed"
        logger.error(f"Failed to deploy model {model_name}: {str(e)}")
        raise e
    
    # Return deployment information
    deployment_info = {
        "model_name": model_name,
        "docker_image": docker_image_tag,
        "namespace": namespace,
        "status": deployment_status,
        "service_port": 3000,
        "configurations": k8s_configs,
        "url": "chat-rag.staging.cloudinfra.zenml.io",
    }
    
    if services:
        bentoml_deployment= cast(BentoMLLocalDeploymentService, services[0])
        zenml_client.update_service(
            id=bentoml_deployment.uuid,
            prediction_url="https://chat-rag.staging.cloudinfra.zenml.io",
            health_check_url="https://chat-rag.staging.cloudinfra.zenml.io/healthz",
            labels={
                "docker_image": docker_image_tag,
                "namespace": namespace,
            }
        )
    
    return deployment_info



def sanitize_name(name: str) -> str:
    # Convert to lowercase and replace invalid characters with '-'
    sanitized = re.sub(r"[^a-z0-9-]", "-", name.lower())
    # Trim to a maximum length of 63 characters and strip leading/trailing '-'
    sanitized = sanitized[:63].strip("-")
    # Ensure the name doesn't start or end with '-'
    sanitized = sanitized.strip("-")
    return sanitized