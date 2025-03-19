# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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

from utils.config_loaders import load_config
from zenml import Model


def get_zenml_project() -> Model:
    """Get ZenML project configuration from pipeline config.

    Args:
        config: Pipeline configuration containing project info

    Returns:
        Model: ZenML project configuration
    """
    config = load_config()

    return Model(
        name=config["project"]["name"],
        version=config["project"]["version"],
        description=config["project"]["description"],
        tags=config["project"]["tags"],
    )


zenml_project = get_zenml_project()
