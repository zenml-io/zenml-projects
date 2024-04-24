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

from PIL import Image
from zenml.io import fileio

from materializers.label_studio_yolo_dataset_materializer import (
    LabelStudioYOLODataset,
)
from utils.split_data import generate_yaml, split_dataset, unzip_dataset


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
