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
from zenml.logger import get_logger

from utils.dataset_utils import split_large_image, convert_bbox_for_ls, \
    export_to_gcp

Image.MAX_IMAGE_PIXELS = None


logger = get_logger(__name__)


@step
def download_and_tile_dataset_from_hf(dataset: str, data_source: str, max_tile_size:int = 1000) -> Dict[str, Any]:
    # Load dataset from huggingface
    dataset = load_dataset(dataset)
    data = dataset["train"]

    # Create local directory to initially copy data to
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_images = {}

    # iterate through the dataset
    for i, d in enumerate(data):
        img = d['image']
        img_name = f"image_{i}"

        width, height = d['image'].size
        # in case images are very large, we split them up into 4 smaller tiles
        if width > max_tile_size or height > max_tile_size:
            logger.info(f"Image {img_name} is larger than {max_tile_size} pixels in "
                        f"at least one dimension, we'll cut it down into "
                        f"separate tiles.")
            split_large_image(
                d=d,
                img_name=img_name,
                output_dir=output_dir,
                all_images=all_images,
                data_source=data_source,
                max_tile_size=max_tile_size
            )
        else:
            img_path = f'{output_dir}/{img_name}.png'

            export_to_gcp(
                data_source=data_source,
                img=img,
                img_name=img_name,
                img_path=img_path
            )

            results = []

            print(d['objects']['bbox'])
            for j, bbox in enumerate(d['objects']['bbox']):
                x1, y1, x2, y2 = bbox
                results.append(
                    convert_bbox_for_ls(
                        x1=x1, x2=x2, y1=y1, y2=y2, width=width,height=height
                    )
                )

            all_images[img_path] = results

    return all_images
