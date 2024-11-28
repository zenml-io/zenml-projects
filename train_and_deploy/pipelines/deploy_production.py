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

from steps import dockerize_bento_model, notify_on_failure, notify_on_success, deploy_model_to_k8s

from zenml import pipeline


@pipeline(on_failure=notify_on_failure, enable_cache=False)
def gitguarden_production_deployment(
    target_env: str,
):
    """Model deployment pipeline.

    This is a pipeline deploys trained model for future inference.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    ########## Deployment stage ##########
    # Get the production model artifact
    bento_model_image = dockerize_bento_model(target_env=target_env)
    deploy_model_to_k8s(bento_model_image)

    notify_on_success(after=["deploy_model_to_k8s"])
    ### YOUR CODE ENDS HERE ###
