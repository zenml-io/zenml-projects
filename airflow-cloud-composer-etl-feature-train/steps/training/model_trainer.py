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

from typing import Dict, Tuple, Union

import xgboost as xgb
from materializers import BigQueryDataset, CSVDataset
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def train_xgboost_model(
    dataset: Union[BigQueryDataset, CSVDataset],
) -> Tuple[
    Annotated[
        xgb.Booster, ArtifactConfig(name="xgb_model", artifact_type=ArtifactType.MODEL)
    ],
    Annotated[Dict[str, float], "metrics"],
]:
    """Train an XGBoost model on the given data.

    Args:
        df: Dataframe containing the data.

    Returns:
        Tuple[xgb.Booster, Dict[str, float]]: Trained model and metrics.
    """
    df = dataset.read_data()
    X = df[["augmented_rate", "rate_diff"]]
    y = df["main_refinancing_operations"]

    dtrain = xgb.DMatrix(X, label=y)
    params = {"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror"}
    model = xgb.train(params, dtrain, num_boost_round=100)

    predictions = model.predict(dtrain)
    mse = ((predictions - y) ** 2).mean()
    r2 = 1 - (((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    return model, {"mse": float(mse), "r2": float(r2)}
