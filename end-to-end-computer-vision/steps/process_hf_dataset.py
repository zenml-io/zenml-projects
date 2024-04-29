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

from utils.dataset_utils import split_image_into_tiles

Image.MAX_IMAGE_PIXELS = None


logger = get_logger(__name__)


@step
def process_hf_dataset(
    dataset: str, data_source: str, max_tile_size: int = 1000
) -> Dict[str, Any]:
    """Downloads a hf dataset and does some processing.

    Converts the labels into the label_studio format.
    Also uploads the images to the datasource path.

    The return of this step maps the path of images in the local fs to the
    annotations in a format that label-studio can import.
    {'data/image_116.png': [
        {'original_width': 907,
        'original_height': 937,
        'image_rotation': 0,
        'value': {
            'x': 66.26240352811466,
            'y': 77.5880469583778,
            'width': 5.071664829106946,
            'height': 3.4151547491995733,
            'rotation': 0,
            'rectanglelabels': ['ship']
        },
        'from_name': 'label',
        'to_name': 'image',
        'type': 'rectanglelabels',
        'origin': 'manual'}, ...
    ]

    Args:
        dataset: Name of the hf dataset
        data_source: Location of the image files
        max_tile_size

    Returns:
        Dictionary mapping filename to bboxes. You can find an example of this
            dict in the docstring above
    """
    # Load dataset from huggingface
    dataset = load_dataset(dataset)
    data = dataset["train"]

    # Create local directory to initially copy data to
    local_output_dir = "data"
    if not os.path.exists(local_output_dir):
        os.mkdir(local_output_dir)

    # Dictionary that maps label information onto the corresponding image
    all_images = {}

    # iterate through the dataset
    for i, d in enumerate(data):
        img_name = f"image_{i}"

        # in case images are very large, we split them up into 4 smaller tiles
        split_image_into_tiles(
            d=d,
            img_name=img_name,
            output_dir=local_output_dir,
            all_images=all_images,
            data_source=data_source,
            max_tile_size=max_tile_size,
        )

    return all_images
