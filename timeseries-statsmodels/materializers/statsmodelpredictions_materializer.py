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
"""Implementation of a materializer for storing predictions."""

import os
from typing import Dict, Union

import pandas as pd
from zenml.enums import VisualizationType
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.pandas_materializer import PandasMaterializer
from zenml.metadata.metadata_types import MetadataType
import matplotlib.pyplot as plt


logger = get_logger(__name__)


class StatsmodelPredictionMaterializer(PandasMaterializer):
    """Statsmodel predictions materializer."""

    def save_visualizations(
        self, df: Union[pd.DataFrame, pd.Series]
    ) -> Dict[str, VisualizationType]:
        """Save visualizations of the given pandas dataframe or series.

        Args:
            df: The pandas dataframe or series to visualize.

        Returns:
            A dictionary of visualization URIs and their types.
        """
        # Save image 
        
        data_df = df["DATA"]
        pred = df["PRED"] 
        
        image_uri = os.path.join(self.uri, "plot.png")
        plt.figure(figsize=(10, 6))
        plt.plot(data_df.index, data_df.values, label='Historical')
        plt.plot(pd.date_range(data_df.index[-1], periods=289)[1:], pred, label='Predicted')
        plt.title('Airline Passengers (5-minute intervals)')
        plt.legend()
        plt.show()
        
        with fileio.open(image_uri, mode="wb") as f:
            plt.savefig(f)

        predictions_path = os.path.join(self.uri, "predictions.csv")
        predictions_path = predictions_path.replace("\\", "/")
        with fileio.open(predictions_path, mode="wb") as f:
            pred.to_csv(f)
        return {
            predictions_path: VisualizationType.CSV,
            image_uri: VisualizationType.IMAGE
        }

    def extract_metadata(
        self, df: Union[pd.DataFrame, pd.Series]
    ) -> Dict[str, "MetadataType"]:
        """Extract metadata from the given pandas dataframe or series.

        Args:
            df: The pandas dataframe or series to extract metadata from.

        Returns:
            The extracted metadata as a dictionary.
        """
        # Call parent method
        pandas_metadata: Dict[str, "MetadataType"] = super().extract_metadata(
            df
        )

        # Add more visualizatons
        
        return pandas_metadata
