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

"""Configuration constants."""

import os
from pathlib import Path
from typing import Dict

# Keep directory configurations, Modal settings, and incident reporting as classes
# since they don't benefit as much from being enums


class Directories:
    """Required local directory paths."""

    RISK = "docs/risk"
    RELEASES = "docs/releases"
    TEMPLATES = "docs/templates"
    SAMPLE_INPUTS_PATH = f"{TEMPLATES}/sample_inputs.json"
    RISK_REGISTER_PATH = f"{RISK}/risk_register.xlsx"

    @classmethod
    def ensure_exist(cls) -> None:
        """Ensure minimal local directories exist."""
        for dir_path in [cls.RISK, cls.RELEASES, cls.TEMPLATES]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class ModalConfig:
    """Modal configuration parameters."""

    VOLUME_NAME = "credit-scoring"
    ENVIRONMENT = "main"

    @classmethod
    def get_volume_metadata(cls) -> Dict[str, str]:
        """Return Modal volume metadata as a dictionary."""
        return {
            "volume_name": cls.VOLUME_NAME,
            "secret_name": "credit-scoring-secrets",
            "app_name": "credit-scoring-app",
            "environment_name": cls.ENVIRONMENT,
            "model_path": "models/model.pkl",
            "preprocess_pipeline_path": "pipelines/preprocess_pipeline.pkl",
            "docs_dir": "docs",
            "risk_register_path": f"{Directories.RISK}/risk_register.xlsx",
            "incident_log_path": f"{Directories.RISK}/incident_log.json",
            "releases_dir": Directories.RELEASES,
            "templates_dir": Directories.TEMPLATES,
            "risk_dir": Directories.RISK,
        }


class Incident:
    """Incident reporting configuration."""

    SLACK_CHANNEL = "#credit-scoring-alerts"
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")


# Initialize required directories at module import
Directories.ensure_exist()
