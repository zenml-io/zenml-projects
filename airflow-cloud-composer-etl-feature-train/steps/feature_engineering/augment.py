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

from datetime import datetime, timezone

import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Augment the data with additional features.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Augmented dataframe.
    """
    logger.info("Augmenting data...")
    df["augmented_rate"] = (
        df[
            "Main refinancing operations - Minimum bid rate/fixed rate (date of changes) - Level (FM.D.U2.EUR.4F.KR.MRR_RT.LEV)"
        ]
        * 2
    )
    df["rate_diff"] = (
        df[
            "Marginal lending facility - date of changes (raw data) - Level (FM.D.U2.EUR.4F.KR.MLFR.LEV)"
        ]
        - df[
            "Deposit facility - date of changes (raw data) - Level (FM.D.U2.EUR.4F.KR.DFR.LEV)"
        ]
    )
    df["augment_timestamp"] = datetime.now(timezone.utc).isoformat()
    logger.info("Data augmentation complete.")
    return df
