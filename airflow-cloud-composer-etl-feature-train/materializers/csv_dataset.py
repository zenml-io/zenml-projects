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

from typing import Optional

import pandas as pd

from materializers.dataset import Dataset


class CSVDataset(Dataset):
    def __init__(self, data_path: str, df: Optional[pd.DataFrame] = None):
        self.data_path = data_path
        if df is not None:
            self.df = df
        else:
            self.df = self.read_data()

    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)
