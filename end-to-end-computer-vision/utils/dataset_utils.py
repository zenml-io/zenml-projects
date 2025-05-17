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
from typing import Any, Dict, List, Tuple

import numpy as np
from materializers.label_studio_export_materializer import (
    LabelStudioAnnotationExport,
)
from PIL import Image
from zenml.io import fileio
from zenml.logger import get_logger

from utils.split_data import generate_yaml, split_dataset, unzip_dataset

logger = get_logger(__name__)


def load_images_from_folder(folder: str) -> List[Image.Image]:
    images: List[Image.Image] = []
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


def load_images_from_source(
    data_source: str, download_dir: str, filenames: List[str]
) -> None:
    total_images = len(filenames)
    for index, filename in enumerate(filenames):
        src_path = f"{data_source}/{filename}.png"
        dst_path = os.path.join(download_dir, f"{filename}.png")
        if not os.path.exists(dst_path):
            fileio.copy(src_path, dst_path)

        if (index + 1) % 100 == 0 or index == total_images - 1:
            logger.info(
                f"{index + 1} of {total_images} images have been downloaded..."
            )


def load_and_split_data(
    dataset: LabelStudioAnnotationExport, data_source: str
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

    # Download images from source bucket and if successful keep them to reuse for future runs
    load_images = False
    download_dir = os.path.join(
        os.getcwd(), "images"
    )  # Temporary dirname that represents a still incomplete download
    loaded_images = os.path.join(
        os.getcwd(), "loaded-images"
    )  # The dirname used once the download fully completes
    images_folder = os.path.join(
        extract_location, "images"
    )  # tmp dirpath used for the current run only

    # Check that images have not already been downloaded
    if not os.path.exists(loaded_images):
        os.makedirs(download_dir, exist_ok=True)
        load_images = True

    # Checks that new images have not been added since previous download
    if os.path.exists(loaded_images):
        if len(os.listdir(loaded_images)) != len(filenames):
            download_dir = loaded_images
            load_images = True

    if load_images:
        logger.info(f"Downloading images from {data_source}")
        load_images_from_source(data_source, download_dir, filenames)
        os.rename(download_dir, loaded_images)

    os.makedirs(images_folder, exist_ok=True)

    logger.info(f"Copy images to {images_folder}")
    load_images_from_source(loaded_images, images_folder, filenames)

    split_dataset(extract_location, ratio=(0.7, 0.15, 0.15), seed=42)
    yaml_path = generate_yaml(extract_location)
    return yaml_path


def convert_bbox_for_ls(x1, x2, y1, y2, width, height) -> Dict[str, Any]:
    """This function converts the bounding box coordinates to the label studio format.

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


def crop_and_adjust_bboxes(
    image: np.array, bboxes: List[List[int]], crop_coordinates: Tuple[int]
):
    """Crops an image and adjust the bboxes that are within the croped portion.

    Args:
        image: Image as np.array
        bboxes: List of bboxes [[x1, y1, x2, y2], [x1, y1, x2, y2]]
        crop_coordinates: Coordinates of the crop, format (x1, y1, x2, y2)

    Returns:
        Tuple containing the cropped portion of the images and all bboxes
            within the crop area
    """
    x_crop_min, y_crop_min, x_crop_max, y_crop_max = crop_coordinates
    cropped_image = image[y_crop_min:y_crop_max, x_crop_min:x_crop_max]

    adjusted_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        # Adjust bounding box coordinates
        xmin_new = max(0, xmin - x_crop_min)
        ymin_new = max(0, ymin - y_crop_min)
        xmax_new = min(x_crop_max - x_crop_min, xmax - x_crop_min)
        ymax_new = min(y_crop_max - y_crop_min, ymax - y_crop_min)

        # Check if the adjusted bounding box is still within the cropping area
        if xmin_new < xmax_new and ymin_new < ymax_new:
            adjusted_bboxes.append([xmin_new, ymin_new, xmax_new, ymax_new])

    return cropped_image, adjusted_bboxes


def split_image_into_tiles(
    d: Any,
    img_name: str,
    output_dir: str,
    all_images: Dict[str, Any],
    data_source: str,
    max_tile_size: int = 500,
):
    """Some images are too large to be useful, this splits them into multiple tiles.

    Args:
        d: One hf dataset entry - needs to contain d['image'] and
            d['objects']['bbox']
        img_name: Name of the image (excluding any suffix)
        output_dir: Where to locally save the image
        all_images: Dictionary containing the image_name<->bboxes mapping
        max_tile_size: Maximum tile size
    """
    tile_id = 0
    img = d["image"]
    bboxes = d["objects"]["bbox"]
    width, height = d["image"].size

    logger.info(f"Processing {img_name} ...")
    for x in range(0, width, max_tile_size):
        print(f"increased x={x}")
        for y in range(0, height, max_tile_size):
            print(f"increased y={y}")
            if x + max_tile_size <= width:
                x1 = x
                x2 = min(x + max_tile_size, width)
            else:
                # The last tile of the row also stays 500 pixels wide by
                #  cropping from max width to the left
                x1 = max(0, width - max_tile_size)
                x2 = width
            if y + max_tile_size <= height:
                y1 = y
                y2 = min(y + max_tile_size, height)
            else:
                # The last tile of the row also stays 500 pixels high by
                #  cropping from max height up
                y1 = max(0, height - max_tile_size)
                y2 = height
            cropped_img, adjusted_bboxes = crop_and_adjust_bboxes(
                image=np.array(img),
                bboxes=bboxes,
                crop_coordinates=(x1, y1, x2, y2),
            )

            # store this tile
            new_img_name = img_name + "_" + str(tile_id)
            img_path = f"{output_dir}/{new_img_name}.png"

            print(f"Storing tile {tile_id} of {img_name} at {img_path} ...")
            export_to_gcp(
                data_source=data_source,
                img=Image.fromarray(cropped_img),
                img_name=new_img_name,
                img_path=img_path,
            )

            logger.info(f"Calculating new bboxes for tile {img_path}")
            tile_width = x2 - x1
            tile_height = y2 - y1

            results = []
            for bbox in adjusted_bboxes:
                results.append(
                    convert_bbox_for_ls(
                        x1=bbox[0],
                        x2=bbox[2],
                        y1=bbox[1],
                        y2=bbox[3],
                        width=tile_width,
                        height=tile_height,
                    )
                )

            all_images[f"{new_img_name}.png"] = results
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
