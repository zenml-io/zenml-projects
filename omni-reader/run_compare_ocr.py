"""Module for running OCR comparison without using ZenML pipeline."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional

# For faster performance in interactive mode without ZenML overhead,
# we implement the OCR functions directly here
import instructor
import polars as pl
from dotenv import load_dotenv
from litellm import completion
from mistralai import Mistral
from PIL import Image

from schemas.image_description import ImageDescription
from utils.encode_image import encode_image
from utils.metrics import compare_results
from utils.prompt import get_prompt

load_dotenv()


def run_gemma3_ocr_direct(
    image: str | Image.Image,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract text directly using gemma3 model.

    Args:
        image: Path to image or PIL image
        custom_prompt: Optional custom prompt

    Returns:
        Dict with extraction results
    """
    start_time = time.time()
    content_type, image_base64 = encode_image(image)

    client = instructor.from_litellm(completion)
    model_name = "ollama/gemma3:27b"

    prompt = custom_prompt if custom_prompt else get_prompt()

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

        return result
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        return {
            "raw_text": "Error: Failed to extract text",
            "description": "Error: Failed to extract description",
            "entities": [],
            "error": error_message,
            "processing_time": time.time() - start_time,
            "model": model_name,
        }


def run_mistral_ocr_direct(
    image: str | Image.Image,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract text directly using mistral model.

    Args:
        image: Path to image or PIL image
        custom_prompt: Optional custom prompt

    Returns:
        Dict with extraction results
    """
    start_time = time.time()
    content_type, image_base64 = encode_image(image)

    mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    client = instructor.from_mistral(mistral_client)

    model_name = "pixtral-12b-2409"

    prompt = custom_prompt if custom_prompt else get_prompt()

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

        print(f"Response: {response}")

        processing_time = time.time() - start_time

        result = {
            "raw_text": response.raw_text if response.raw_text else "No text found",
            "description": response.description if response.description else "No description found",
            "entities": response.entities if response.entities else [],
            "processing_time": processing_time,
            "model": model_name,
        }

        return result
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        return {
            "raw_text": "Error: Failed to extract text",
            "description": "Error: Failed to extract description",
            "entities": [],
            "error": error_message,
            "processing_time": time.time() - start_time,
            "model": model_name,
        }


def run_ocr(
    image: str | Image.Image,
    model: str = "gemma3",
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run OCR using either Gemma3 or Mistral model.

    Args:
        image: Path to image or PIL image
        model: Model to use ('gemma3' or 'mistral')
        custom_prompt: Optional custom prompt

    Returns:
        Dict with extraction results
    """
    if model.lower() == "gemma3":
        return run_gemma3_ocr_direct(image=image, custom_prompt=custom_prompt)
    else:
        return run_mistral_ocr_direct(image=image, custom_prompt=custom_prompt)


def compare_models(
    image_paths: List[str],
    custom_prompt: Optional[str] = None,
    ground_truth_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare Gemma3 and Mistral OCR capabilities on a list of images.

    Args:
        image_paths: List of paths to images
        custom_prompt: Optional custom prompt to use for both models
        ground_truth_texts: Optional list of ground truth texts
    Returns:
        Dictionary with comparison results
    """
    results = {
        "gemma_results": [],
        "mistral_results": [],
        "ground_truth": [],
    }

    for i, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)

        print(f"Processing image {i + 1}/{len(image_paths)}: {image_name}")

        # Run both models
        gemma_result = run_ocr(
            image=image_path,
            model="gemma3",
            custom_prompt=custom_prompt,
        )
        mistral_result = run_ocr(
            image=image_path,
            model="mistral",
            custom_prompt=custom_prompt,
        )

        # Create entries for dataframes
        gemma_entry = {
            "id": i,
            "image_name": image_name,
            "gemma_text": gemma_result["raw_text"],
            "gemma_entities": ", ".join(gemma_result.get("entities", [])),
            "gemma_processing_time": gemma_result.get("processing_time", 0),
        }

        mistral_entry = {
            "id": i,
            "image_name": image_name,
            "mistral_text": mistral_result["raw_text"],
            "mistral_entities": ", ".join(mistral_result.get("entities", [])),
            "mistral_processing_time": mistral_result.get("processing_time", 0),
        }

        results["gemma_results"].append(gemma_entry)
        results["mistral_results"].append(mistral_entry)

        # Add ground truth if available
        if ground_truth_texts and i < len(ground_truth_texts):
            results["ground_truth"].append(
                {
                    "id": i,
                    "image_name": image_name,
                    "ground_truth_text": ground_truth_texts[i],
                }
            )

            # Calculate metrics
            metrics = compare_results(
                ground_truth_texts[i],
                gemma_result["raw_text"],
                mistral_result["raw_text"],
            )
            print(f"Metrics for {image_name}:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare OCR models")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        help="Model to use: 'gemma3', 'mistral', or 'both'",
    )
    parser.add_argument("--prompt", type=str, help="Custom prompt to use")

    args = parser.parse_args()

    if args.model.lower() == "both":
        start_time = time.time()
        gemma_result = run_ocr(args.image, "gemma3", args.prompt)
        mistral_result = run_ocr(args.image, "mistral", args.prompt)
        print("\nGemma3 results:")
        print(f"Text: {gemma_result['raw_text']}")
        print(f"Entities: {gemma_result.get('entities', [])}")
        print(f"Processing time: {gemma_result.get('processing_time', 0):.2f}s")

        print("\nMistral results:")
        print(f"Text: {mistral_result['raw_text']}")
        print(f"Entities: {mistral_result.get('entities', [])}")
        print(f"Processing time: {mistral_result.get('processing_time', 0):.2f}s")

        print(f"\nTotal time: {time.time() - start_time:.2f}s")
    else:
        result = run_ocr(args.image, args.model, args.prompt)
        print(f"\n{args.model} results:")
        print(f"Text: {result['raw_text']}")
        print(f"Entities: {result.get('entities', [])}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
