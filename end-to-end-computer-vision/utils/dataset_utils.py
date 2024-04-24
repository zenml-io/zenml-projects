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
import tempfile
from typing import Dict, Any

import numpy as np
from PIL import Image
from zenml.io import fileio

from materializers.label_studio_yolo_dataset_materializer import (
    LabelStudioYOLODataset,
)
from utils.split_data import generate_yaml, split_dataset, unzip_dataset
from zenml.logger import get_logger

logger = get_logger(__name__)


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        ):
            with open(os.path.join(folder, filename), "rb") as f:
                img = Image.open(f)
                images.append(img)
    return images


def load_and_split_data(
    dataset: LabelStudioYOLODataset, data_source: str
) -> str:
    """Load data from dataset into file system and split into train/val.

    First download the .zip file containing the label.txt files from the zenml
    Model. Then downloads the corresponding images from the data source.

    Args:
        dataset: Dataset object that points to the zip file that contains the
            label files.
        data_source: Path of the remote data source that contains the images.
    """
    tmpfile_ = tempfile.NamedTemporaryFile(dir="data", delete=False)
    tmpdirname = os.path.basename(tmpfile_.name)
    extract_location = os.path.join(tmpdirname, "data")
    unzip_dataset(dataset.filepath, extract_location)

    # Get all filenames inside the labels subfolder
    labels_folder = os.path.join(extract_location, "labels")
    filenames = [
        os.path.splitext(f)[0]
        for f in os.listdir(labels_folder)
        if f.endswith(".txt")
    ]

    # Download corresponding images from gcp bucket
    images_folder = os.path.join(extract_location, "images")
    os.makedirs(images_folder, exist_ok=True)

    for filename in filenames:
        src_path = f"{data_source}/{filename}.png"
        dst_path = os.path.join(images_folder, f"{filename}.png")
        fileio.copy(src_path, dst_path)

    split_dataset(extract_location, ratio=(0.7, 0.15, 0.15), seed=42)
    return generate_yaml(extract_location)


def convert_bbox_for_ls(x1, x2, y1, y2, width, height) -> Dict[str, Any]:
    """ This function converts the bounding box coordinates to the label studio format.

    Parameters:
    x1, x2, y1, y2 (int): The initial bounding box coordinates in pixels.
    width, height (int): The original dimensions of the image.

    Returns:
    A dictionary approximating the label studio.
    """
    x = x1 / width
    y = y1 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    print(f"Converting bbox to results.")
    return {
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": x*100,
                "y": y*100,
                "width": w*100,
                "height": h*100,
                "rotation": 0,
                "rectanglelabels": [
                    "ship"
                ]
            },
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual"
        }


def split_large_image(
    d: Any, img_name: str, output_dir: str, all_images: Dict[str, Any], data_source: str
):
    """Some images are too large to be useful, this splits them into multiple tiles.

    Args:
        d: The hf dataset entry
        img_name: Name of the image
        output_dir: Where to initially save the image to
        all_images: Dictionary containing the image_name<->bboxes mapping
    """
    tile_id = 0
    img = d['image']
    width, height = d['image'].size

    img_tiles = [img.crop((x, y, x + width // 2, y + height // 2))
                 for x in range(0, width, width // 2)
                 for y in range(0, height, height // 2)]

    print(f"Split {img_name} into 4 tiles.")

    # Iterate through all 4 tiles, save the image and calculate the new bboxes
    for i, img_tile in enumerate(img_tiles):
        # Create new dictionary for each tile
        new_img_name = img_name + '_' + str(tile_id)
        img_path = f'{output_dir}/{new_img_name}.png'

        export_to_gcp(
            data_source=data_source,
            img=img,
            img_name=new_img_name,
            img_path=img_path
        )

        print(f"{img_path} saved.")
        results = []
        for j, bbox in enumerate(d["objects"]["bbox"]):
            x1, y1, x2, y2 = np.clip(bbox, [i * (width // 2), i * (height // 2),
                                            (i + 1) * (width // 2),
                                            (i + 1) * (height // 2)],
                                     [0, 0, width // 2, height // 2])
            results.append(
                convert_bbox_for_ls(
                    x1=x1, x2=x2, y1=y1, y2=y2, width=width, height=height
                )
            )

        all_images[img_path] = results
        tile_id += 1


def export_to_gcp(data_source: str, img: Image, img_name: str, img_path: str):
    """
    Saves a given image locally, then copies it to a Google Cloud Storage bucket.

    Parameters:
    data_source: The name of source directory or bucket where image needs to be
        stored.
    img: The image to be stored. This should be an instance of the PIL Image
        class.
    img_name: The name to be used when saving the image in the GCP bucket. Should not contain '.png'
    img_path: The local path where the image will be temporarily saved before
        being copied to the bucket.
    """
    logger.info(f"Storing image to {img_path}.")
    img.save(img_path)
    bucket_path = os.path.join(data_source, f"{img_name}.png")
    logger.info(f"Copying into gcp bucket {bucket_path}")
    fileio.copy(img_path, bucket_path, overwrite=True)
