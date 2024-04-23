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
import os
from typing import Any, Dict

from datasets import load_dataset
from PIL import Image
from zenml import step
from zenml.io import fileio
from zenml.logger import get_logger

Image.MAX_IMAGE_PIXELS = None

logger = get_logger(__name__)


@step(enable_cache=True)
def download_dataset_from_hf(dataset: str, data_source: str) -> Dict[str, Any]:
    dataset = load_dataset(dataset)
    data = dataset['train']

    output_dir = "data"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_images = {}

    for i, d in enumerate(data):
        img = d['image']
        img_name = f"image_{i}.png"
        img_path = f'{output_dir}/{img_name}'

        logger.info(f"Storing image to {img_path}.")
        img.save(img_path)

        bucket_path = os.path.join(data_source, img_name)
        logger.info(f"Copying into gcp bucket {bucket_path}")
        fileio.copy(img_path, bucket_path, overwrite=True)

        width, height = d['image'].size

        results = []
        for j, bbox in enumerate(d["objects"]["bbox"]):
            x1, y1, x2, y2 = bbox
            x = x1 / width
            y = y1 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            results.append(
                {
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": x * 100,
                        "y": y * 100,
                        "width": w * 100,
                        "height": h * 100,
                        "rotation": 0,
                        "rectanglelabels": ["ship"],
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": "manual",
                }
            )

        all_images[img_name] = results
        if i > 20:
            break
    return all_images
