# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from steps.bento_dockerizer import bento_dockerizer
from steps.k8s_deployment import k8s_deployment
from steps.visualize_chat import create_chat_interface
from zenml import pipeline


@pipeline(enable_cache=False)
def zenml_docs_chatbot_deployer():
    """Model deployment pipeline.

    This is a pipeline deploys trained model for future inference.
    """
    bento_model_image = bento_dockerizer()
    deployment_info = k8s_deployment(bento_model_image)
    create_chat_interface(deployment_info)
