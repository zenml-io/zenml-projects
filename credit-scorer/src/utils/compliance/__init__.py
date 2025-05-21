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

"""Utility functions for the compliance module."""

from .compliance_constants import (
    ARTICLE_DESCRIPTIONS,
    COMPLIANCE_DATA_SOURCES,
    DEFAULT_COMPLIANCE_PATHS,
    EU_AI_ACT_ARTICLES,
)
from .exceptions import (
    ComplianceCalculationError,
    ComplianceDataError,
    ComplianceError,
)
from .schemas import (
    ComplianceThresholds,
    EUArticle,
    RiskCategory,
    RiskStatus,
)
