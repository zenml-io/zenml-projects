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
from pathlib import Path
from typing import Dict, Optional
import re
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

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

@step
def k8s_deployment(
    docker_image_tag: str,
    namespace: str = "default"
) -> Dict:
    # Get the raw model name
    raw_model_name = get_step_context().model.name
    # Sanitize the model name
    model_name = sanitize_name(raw_model_name)
    
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
            
        elif config["kind"] == "Deployment":
            # Update deployment selector and template
            config["spec"]["selector"]["matchLabels"]["app"] = model_name
            config["spec"]["template"]["metadata"]["labels"]["app"] = model_name
            
            # Update the container image and name
            containers = config["spec"]["template"]["spec"]["containers"]
            for container in containers:
                container["name"] = model_name
                container["image"] = docker_image_tag
    
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
        "configurations": k8s_configs
    }
    
    return deployment_info



def sanitize_name(name: str) -> str:
    # Convert to lowercase and replace invalid characters with '-'
    sanitized = re.sub(r"[^a-z0-9-]", "-", name.lower())
    # Trim to a maximum length of 63 characters and strip leading/trailing '-'
    sanitized = sanitized[:63].strip("-")
    # Ensure the name doesn't start or end with '-'
    sanitized = sanitized.strip("-")
    return sanitized