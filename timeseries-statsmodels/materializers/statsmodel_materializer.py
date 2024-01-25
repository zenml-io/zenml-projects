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
"""Implementation of ZenML's pickle materializer."""

import os
from typing import Any, ClassVar, Tuple, Type


from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.logger import get_logger

from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
import os
from typing import Dict

import pandas as pd
from zenml.enums import VisualizationType
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.cloudpickle_materializer import CloudpickleMaterializer
from zenml.metadata.metadata_types import MetadataType


logger = get_logger(__name__)


class StatsmodelMaterializer(CloudpickleMaterializer):
    """Statsmodel materializer."""

    ASSOCIATED_TYPES: ClassVar[Tuple[Type[Any], ...]] = (SARIMAXResultsWrapper,)
    ASSOCIATED_ARTIFACT_TYPE: ClassVar[ArtifactType] = ArtifactType.MODEL

    def save_visualizations(
        self, fitted_model: SARIMAXResultsWrapper
    ) -> Dict[str, VisualizationType]:
        """Save visualizations of the given pandas dataframe or series.

        Args:
            fitted_model: The pandas dataframe or series to visualize.

        Returns:
            A dictionary of visualization URIs and their types.
        """
        describe_uri = os.path.join(self.uri, "describe.csv")
        describe_uri = describe_uri.replace("\\", "/")
        with fileio.open(describe_uri, mode="wb") as f:
            f.write(fitted_model.summary().as_csv().encode())
        return {describe_uri: VisualizationType.CSV}
