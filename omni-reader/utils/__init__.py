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

from .encode_image import encode_image
from .metrics import compare_multi_model
from .ocr_processing import (
    log_image_metadata,
    log_error_metadata,
    log_summary_metadata,
    process_images_with_model,
    process_image,
)
from .prompt import (
    get_prompt,
    ImageDescription,
)
from .io_utils import (
    save_ocr_data_to_json,
    load_ocr_data_from_json,
    list_available_ground_truth_files,
)
from .model_configs import (
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    get_model_info,
    model_registry,
    ModelConfig,
    get_model_prefix,
)
from .extract_json import try_extract_json_from_response
