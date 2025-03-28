"""This module contains the steps for running the mistral OCR model."""

import logging
import os
import time
from typing import Optional

import instructor
from dotenv import load_dotenv
from mistralai import Mistral
from PIL import Image

# from zenml import log_metadata, step
from schemas import ImageDescription
from utils import (
    encode_image,
    get_prompt,
)

load_dotenv()


# @step(enable_cache=False)
def run_mistral_ocr(
    image: Image.Image | str,
    custom_prompt: Optional[str] = None,
) -> dict:
    """Extract text and identify entities in an image using mistral model.

    Args:
        image: Either a PIL Image object or a string path to an image file
        custom_prompt: Optional custom prompt to override the default prompt

    Returns:
        dict: Structured extraction results with text and entities
    """
    start_time = time.time()
    content_type, image_base64 = encode_image(image)

    mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    client = instructor.from_mistral(mistral_client)

    model_name = "pixtral-12b-2409"

    prompt = custom_prompt if custom_prompt else get_prompt()

    logging.info(f"Running Mistral OCR with model: {model_name}")
    logging.info(f"Using prompt: {prompt}")

    try:
        response = client.chat.completions.create(
            model=model_name,
            response_model=ImageDescription,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:{content_type};base64,{image_base64}",
                        },
                    ],
                }
            ],
        )

        processing_time = time.time() - start_time

        result = {
            "raw_text": response.raw_text if response.raw_text else "No text found",
            "description": response.description if response.description else "No description found",
            "entities": response.entities if response.entities else [],
            "processing_time": processing_time,
            "model": model_name,
        }

        # Log metadata for this step
        # log_metadata(
        #     metadata={
        #         "mistral_ocr": {
        #             "model": model_name,
        #             "processing_time_seconds": processing_time,
        #             "text_length": len(result["raw_text"]),
        #             "entities_count": len(result["entities"]),
        #         }
        #     }
        # )

        # Display results in terminal for debugging
        logging.info(
            f"Mistral OCR results: {len(result['raw_text'])} chars of text, {len(result['entities'])} entities"
        )
        logging.info(f"Processing time: {processing_time:.2f} seconds")

        return result
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logging.error(error_message)

        # Log error in metadata
        # log_metadata(
        #     metadata={
        #         "mistral_ocr_error": {
        #             "error": str(e),
        #             "model": model_name,
        #         }
        #     }
        # )

        return {
            "raw_text": "Error: Failed to extract text",
            "description": "Error: Failed to extract description",
            "entities": [],
            "error": error_message,
            "processing_time": time.time() - start_time,
            "model": model_name,
        }
