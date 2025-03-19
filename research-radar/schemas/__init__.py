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

# ruff: noqa: I001

# schemas that don't depend on other schemas
from .classification_output import (
    ClassificationOutput,
)
from .claude_response import (
    ClaudeResponse,
)

from .input_article import (
    ArticleMeta,
    InputArticle,
)
from .training_config import (
    TrainingConfig,
)

# schemas that may depend on the above
from .config_models import (
    AppConfig,
    BatchProcessingConfig,
    CheckpointConfig,
    ClassificationPipelineConfig,
    DatasetPathsConfig,
    DataSplitConfig,
    InferenceParamsConfig,
    ModelRepoIdsConfig,
    OutputPathsConfig,
    ParallelProcessingConfig,
    ProjectConfig,
    validate_config,
)

# zenml_project depends on utils.config_loaders
# which might import from schemas
from .zenml_project import (
    zenml_project,
)
