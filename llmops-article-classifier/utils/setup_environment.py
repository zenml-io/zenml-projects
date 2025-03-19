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

import os
from functools import wraps

import huggingface_hub
from dotenv import load_dotenv

from materializers.register_materializers import register_materializers
from utils import logger
from zenml.client import Client

load_dotenv()


def get_hf_token():
    """Get the Hugging Face token from the environment or ZenML client."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            logger.info("Authenticating to Hugging Face via ZenML client...")
            client = Client()
            hf_token = client.get_secret("hf_token").secret_values["token"]
        except Exception as e:
            logger.warning(f"Error authenticating with Hugging Face: {e}")
    return hf_token


def setup_environment():
    """Setup environment with required authentications."""

    register_materializers()

    hf_token = get_hf_token()

    huggingface_hub.login(token=hf_token)

    # Disable tokenizer parallelism to avoid deadlocks
    # and to silence the warning of course :D
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def with_setup_environment(f):
    """Decorator to ensure the environment is setup before the command runs."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        setup_environment()
        return f(*args, **kwargs)

    return wrapper
