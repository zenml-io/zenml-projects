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
from zenml.client import Client

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


def load_and_split_data(dataset: LabelStudioYOLODataset) -> str:

    tmpfile_ = tempfile.NamedTemporaryFile(dir="data", delete=False)
    tmpdirname = os.path.basename(tmpfile_.name)

    extract_location = os.path.join(tmpdirname, "data")

    unzip_dataset(dataset.filepath, extract_location)
    split_dataset(extract_location, ratio=(0.7, 0.15, 0.15), seed=42)
    return generate_yaml(extract_location)
