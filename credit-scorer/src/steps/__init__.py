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

"""Steps for the project."""

# deployment
from .deployment.approve import approve_deployment
from .deployment.deploy import modal_deployment
from .deployment.generate_dashboard import generate_compliance_dashboard
from .deployment.generate_sbom import generate_sbom
from .deployment.post_market_monitoring import post_market_monitoring
from .deployment.post_run_annex import generate_annex_iv_documentation

# feature engineering
from .feature_engineering.data_preprocessor import data_preprocessor
from .feature_engineering.data_profiler import data_profiler
from .feature_engineering.data_splitter import data_splitter
from .feature_engineering.ingest import ingest
from .training.evaluate import evaluate_model
from .training.risk_assessment import risk_assessment

# training
from .training.train import train_model
