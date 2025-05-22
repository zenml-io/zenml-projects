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


from datetime import datetime
from typing import Annotated

import pandas as pd
from whylogs.core import DatasetProfileView
from zenml import step
from zenml.integrations.whylogs.data_validators.whylogs_data_validator import (
    WhylogsDataValidator,
)
from zenml.integrations.whylogs.flavors.whylogs_data_validator_flavor import (
    WhylogsDataValidatorSettings,
)

from src.constants import Artifacts as A

whylogs_settings = WhylogsDataValidatorSettings(
    enable_whylabs=False, dataset_id="credit-scorer-model"
)


@step(
    enable_cache=False,
    settings={"data_validator": whylogs_settings},
)
def data_profiler(
    df: pd.DataFrame,
) -> Annotated[DatasetProfileView, A.WHYLOGS_PROFILE]:
    """Generate a WhyLogs profile for the dataset.

    EU AI Act compliance:
    - Article 12 (Record-Keeping): Supports points 2(a-c) by enabling data monitoring,
      detecting quality issues, and facilitating post-market surveillance
    - Article 10 (Data Governance): Provides statistical documentation of data characteristics

    Args:
        df: a Pandas DataFrame

    Returns:
        Whylogs Profile generated for the dataset
    """
    # Method 1: Using WhyLogs data validator
    data_validator = WhylogsDataValidator.get_active_data_validator()
    profile = data_validator.data_profiling(
        df, dataset_timestamp=datetime.now()
    )

    # Method 2: Alternatively, we could use whylogs directly
    # results = why.log(df)
    # profile = results.profile().view()

    return profile
