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
from .extract_json import try_extract_json_from_response
from .metrics import (
    analyze_errors,
    calculate_custom_metrics,
    calculate_model_similarities,
    compare_multi_model,
    find_best_model,
    normalize_text,
)
from .model_configs import (
    DEFAULT_MODEL,
    MODEL_CONFIGS,
    ModelConfig,
    get_model_info,
    get_model_prefix,
    model_registry,
)
from .ocr_processing import (
    log_error_metadata,
    log_image_metadata,
    log_summary_metadata,
    process_image,
    process_images_with_model,
)
from .prompt import (
    ImageDescription,
    get_prompt,
)
from .visualizations import (
    create_comparison_table,
    create_metrics_table,
    create_model_card_with_logo,
    create_model_comparison_card,
    create_model_similarity_matrix,
    create_ocr_batch_visualization,
    create_summary_visualization,
)
