"""Module for running OCR comparison without using ZenML pipeline."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import instructor
from dotenv import load_dotenv
from litellm import completion
from mistralai import Mistral
from PIL import Image

from utils.encode_image import encode_image
from utils.prompt import ImageDescription, get_prompt

load_dotenv()


def run_ocr_from_ui(
    image: str | Image.Image,
    model: str = "ollama/gemma3:27b",
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract text directly using OCR model.

    This function is designed for use in the streamlit app.

    Args:
        image: Path to image or PIL image
        custom_prompt: Optional custom prompt
        model: Name of the model to use
    Returns:
        Dict with extraction results
    """
    start_time = time.time()
    content_type, image_base64 = encode_image(image)

    if "gemma" in model.lower():
        client = instructor.from_litellm(completion)
    elif "mistral" in model.lower() or "pixtral" in model.lower():
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        client = instructor.from_mistral(mistral_client)
    else:
        raise ValueError(f"Unsupported model: {model}")

    prompt = custom_prompt if custom_prompt else get_prompt()

    try:
        response = client.chat.completions.create(
            model=model,
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
            "processing_time": processing_time,
            "model": model,
        }

        return result
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        return {
            "raw_text": "Error: Failed to extract text",
            "error": error_message,
            "processing_time": time.time() - start_time,
            "model": model,
        }


def run_ollama_ocr_from_ui(
    image: str | Image.Image,
    model: str = "gemma3:27b",
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run OCR using Ollama.

    Args:
        image: Path to the image file to process
        model: Name of the model to use
        custom_prompt: Optional custom prompt

    Returns:
        Dict containing OCR results
    """
    import ollama

    from utils.ocr_model_utils import try_extract_json_from_response

    start_time = time.time()

    _, image_base64 = encode_image(image)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": custom_prompt if custom_prompt else get_prompt(),
                    "images": [image_base64],
                    "format": ImageDescription.model_json_schema(),
                }
            ],
        )
        result_json = response.message.content
        processing_time = time.time() - start_time

        result_dict = try_extract_json_from_response(result_json)
        return {
            "raw_text": result_dict.get("raw_text", ""),
            "processing_time": processing_time,
            "model": model,
        }
    except Exception as e:
        print(f"Error with Gemma OCR: {e}")
        return {"raw_text": f"Error: {str(e)}", "success": False}


def compare_models(
    image_paths: List[str],
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare Gemma3 and Mistral OCR capabilities on a list of images.

    Args:
        image_paths: List of paths to images
        custom_prompt: Optional custom prompt to use for both models
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
        gemma_result = run_ocr_from_ui(
            image=image_path,
            model_name="ollama/gemma3:27b",
            custom_prompt=custom_prompt,
        )
        mistral_result = run_ocr_from_ui(
            image=image_path,
            model_name="pixtral-12b-2409",
            custom_prompt=custom_prompt,
        )

        # Create entries for dataframes
        gemma_entry = {
            "id": i,
            "image_name": image_name,
            "gemma_text": gemma_result["raw_text"],
            "gemma_processing_time": gemma_result.get("processing_time", 0),
        }

        mistral_entry = {
            "id": i,
            "image_name": image_name,
            "mistral_text": mistral_result["raw_text"],
            "mistral_processing_time": mistral_result.get("processing_time", 0),
        }

        results["gemma_results"].append(gemma_entry)
        results["mistral_results"].append(mistral_entry)

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
        gemma_result = run_ocr_from_ui(args.image, "ollama/gemma3:27b", args.prompt)
        mistral_result = run_ocr_from_ui(args.image, "pixtral-12b-2409", args.prompt)
        print("\nGemma3 results:")
        print(f"Text: {gemma_result['raw_text']}")
        print(f"Processing time: {gemma_result.get('processing_time', 0):.2f}s")

        print("\nMistral results:")
        print(f"Text: {mistral_result['raw_text']}")
        print(f"Processing time: {mistral_result.get('processing_time', 0):.2f}s")

        print(f"\nTotal time: {time.time() - start_time:.2f}s")
    else:
        result = run_ocr_from_ui(args.image, args.model, args.prompt)
        print(f"\n{args.model} results:")
        print(f"Text: {result['raw_text']}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
