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
"""This module contains utility functions for encoding images to base64 strings."""

import base64
import mimetypes
from io import BytesIO

from PIL import Image


def encode_pil_image(image: Image.Image, format: str = "JPEG") -> str:
    """Encode a PIL Image object to a base64 string.

    Args:
        image: PIL Image object
        format: Image format for encoding (default: JPEG)

    Returns:
        str: Base64 encoded string of the image
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    image_data = buffered.getvalue()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return image_base64


def encode_image_from_path(image_path: str) -> str:
    """Encode an image from a file path to a base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")
    return image_base64


def encode_image(image: Image.Image | str) -> tuple[str, str]:
    """Encode an image to a base64 string.

    Args:
        image: Either a PIL Image object or a string path to an image file

    Returns:
        tuple[str, str]: Image type and base64 encoded string
    """
    if isinstance(image, str):
        content_type = mimetypes.guess_type(image)[0] or "image/jpeg"
        image_base64 = encode_image_from_path(image)
    else:
        img_format = image.format or "JPEG"
        content_type = f"image/{img_format.lower()}" if img_format else "image/jpeg"
        image_base64 = encode_pil_image(image, format=img_format if img_format else "JPEG")

    return content_type, image_base64
