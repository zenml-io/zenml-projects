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

import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_latest_data_local(data_path: str) -> pd.DataFrame:
    """Load the latest data from the data source.

    Args:
        data_path: Path to the data source.

    Returns:
        pd.Datafram: Dataframe containing the data.
    """
    df = pd.read_csv(data_path)
    return df[df["load_timestamp"] == df["load_timestamp"].max()]
