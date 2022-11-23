#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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
from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def yolov5_deployment_pipeline(
    model_loader,
    deployment_trigger,
    bento_builder,
    deployer,
):
    model_path, model = model_loader()
    decision = deployment_trigger(model_path)
    bento = bento_builder(model=model)
    deployer(deploy_decision=decision, bento=bento)
