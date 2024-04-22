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
from zenml.logger import get_logger

from pipelines import data_export
from pipelines.training import training

logger = get_logger(__name__)



if __name__ == "__main__":
    # data_export(dataset_name="cv_proj") #.with_options(config_path="configs/data_export_alexej.yaml")()
    #my_pipeline()
    # data_export.with_options(config_path="configs/data_export_alexej.yaml")()
    training.with_options(config_path="configs/training.yaml")()
