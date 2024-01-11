#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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

import requests
from zenml.integrations.kubeflow.flavors.kubeflow_orchestrator_flavor import (
    KubeflowOrchestratorSettings,
)
from zenml.client import Client


def get_kubeflow_settings() -> dict:
    """Returns the kubeflow settings"""
    orchestrator = Client().active_stack.orchestrator

    if orchestrator.flavor == "kubeflow":
        import os
        NAMESPACE = os.getenv("KUBEFLOW_NAMESPACE")  # This is the user namespace for the profile you want to use
        USERNAME = os.getenv("KUBEFLOW_USERNAME")  # This is the username for the profile you want to use
        PASSWORD = os.getenv("KUBEFLOW_PASSWORD")  # This is the password for the profile you want to use

        def get_kfp_token(username: str, password: str) -> str:
            """Get token for kubeflow authentication."""
            # Resolve host from active stack
            orchestrator = Client().active_stack.orchestrator

            try:
                kubeflow_host = orchestrator.config.kubeflow_hostname
            except AttributeError:
                raise AssertionError(
                    "You must configure the Kubeflow orchestrator "
                    "with the `kubeflow_hostname` parameter which ends "
                    "with `/pipeline` (e.g. `https://mykubeflow.com/pipeline`). "
                    "Please update the current kubeflow orchestrator with: "
                    f"`zenml orchestrator update {orchestrator.name} "
                    "--kubeflow_hostname=<MY_KUBEFLOW_HOST>`"
                )

            session = requests.Session()
            response = session.get(kubeflow_host)
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            data = {"login": username, "password": password}
            session.post(response.url, headers=headers, data=data)
            session_cookie = session.cookies.get_dict()["authservice_session"]
            return session_cookie


        token = get_kfp_token(USERNAME, PASSWORD)
        session_cookie = "authservice_session=" + token
        kubeflow_settings = KubeflowOrchestratorSettings(
            client_args={"cookies": session_cookie}, user_namespace=NAMESPACE
        )
    else:
        kubeflow_settings = {}
        
    return kubeflow_settings
