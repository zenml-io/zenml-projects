"""Module for running OCR comparison from Streamlit app."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from mistralai import Mistral
from openai import OpenAI
from PIL import Image

from utils.encode_image import encode_image
from utils.extract_json import try_extract_json_from_response
from utils.model_configs import (
    DEFAULT_MODEL,
    MODEL_CONFIGS,
)
from utils.ocr_processing import process_image
from utils.prompt import ImageDescription, get_prompt

load_dotenv()


def create_completion(
    client: Mistral | OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
) -> ImageDescription:
    """Create a completion for a given model.

    Args:
        client: The client to use to create the completion
        model_name: The name of the model to use
        messages: The messages to use to create the completion

    Returns:
        The completion as a JSON object (ImageDescription)
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_model=ImageDescription,
    )
    result_json = try_extract_json_from_response(response)
    return result_json


def run_ocr_from_ui(
    image: str | Image.Image,
    model: str,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract text directly using OCR model.

    This function is designed for use in the streamlit app.

    Args:
        image: Path to image or PIL image
        model: ID of the model to use
        custom_prompt: Optional custom prompt

    Returns:
        Dict with extraction results
    """
    start_time = time.time()

    # Get model configuration based on model ID
    if model not in MODEL_CONFIGS:
        return {
            "raw_text": f"Error: Model '{model}' not found in MODEL_CONFIGS",
            "error": f"Invalid model: {model}",
            "processing_time": 0,
            "model": model,
        }

    model_config = MODEL_CONFIGS[model]

    content_type, image_base64 = encode_image(image)

    prompt = custom_prompt if custom_prompt else get_prompt()

    try:
        result_json = process_image(model_config, prompt, image_base64, content_type)
        processing_time = time.time() - start_time

        result = {
            "raw_text": result_json.get("raw_text", "No text found"),
            "processing_time": processing_time,
            "model": model,
            "display_name": model_config.display,
            "provider": model_config.provider,
        }

        for key, value in result_json.items():
            if key not in ["raw_text", "processing_time", "model", "display_name", "provider"]:
                result[key] = value

        return result
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "raw_text": f"Error: Failed to extract text - {str(e)}",
            "error": str(e),
            "processing_time": processing_time,
            "model": model,
            "display_name": model_config.display,
            "provider": model_config.provider,
        }


def run_models_in_parallel(
    image_path: str,
    model_ids: List[str],
    custom_prompt: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Process an image with multiple models in parallel.

    Args:
        image_path: Path to the image file
        model_ids: List of model IDs to process
        custom_prompt: Optional custom prompt

    Returns:
        Dictionary mapping model IDs to their results
    """
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    max_workers = min(len(model_ids), 5)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            model_id: executor.submit(run_ocr_from_ui, image_path, model_id, custom_prompt)
            for model_id in model_ids
        }

        with tqdm(total=len(model_ids), desc="Processing models") as pbar:
            for model_id, future in futures.items():
                try:
                    result = future.result()
                    results[model_id] = result
                except Exception as e:
                    print(f"Error processing model {model_id}: {str(e)}")
                    results[model_id] = {
                        "raw_text": f"Error: {str(e)}",
                        "error": str(e),
                        "processing_time": 0,
                        "model": model_id,
                    }
                finally:
                    pbar.update(1)

    return results


def list_supported_models():
    """List all supported models."""
    print("\nSupported models:")
    print("-" * 70)
    print(f"{'Model ID':<25} {'Display Name':<30} {'Provider':<15}")
    print("-" * 70)

    for model_id, config in MODEL_CONFIGS.items():
        print(f"{model_id:<25} {config.display:<30} {config.provider:<15}")

    print("\nDefault model:", DEFAULT_MODEL.name)
    print("-" * 70)


def format_model_results(model_id, result):
    """Format results for a specific model."""
    model_config = MODEL_CONFIGS.get(model_id, None)
    model_display = model_config.display if model_config else model_id

    output = f"\n{model_display} results:"

    if "error" in result:
        output += f"\n❌ Error: {result.get('error', 'Unknown error')}"
    else:
        text = result["raw_text"]
        if len(text) > 150:
            text = f"{text[:150]}..."
        output += f"\n✅ Text: {text}"

    output += f"\n⏱️ Processing time: {result.get('processing_time', 0):.2f}s"

    if "confidence" in result and result["confidence"] is not None:
        output += f"\n🎯 Confidence: {result['confidence']:.2%}"

    return output


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(description="Compare OCR models")
    parser.add_argument(
        "--image",
        type=str,
        default="assets/street_signs/paris.jpg",
        help="Path to image file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL.name,
        help="Model to use: a specific model ID, 'all' to compare all, or a comma-separated list",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt to use",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported models and exit",
    )

    args = parser.parse_args()

    if args.list:
        list_supported_models()
        return

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    start_time = time.time()

    if args.model.lower() == "all":
        # Run all models in parallel
        print(f"Processing image with all {len(MODEL_CONFIGS)} models in parallel...")

        results = run_models_in_parallel(args.image, list(MODEL_CONFIGS.keys()), args.prompt)

        successful_models = sum(1 for result in results.values() if "error" not in result)
        failed_models = len(results) - successful_models

        print("\n" + "=" * 50)
        print(f"OCR COMPARISON RESULTS ({successful_models} successful, {failed_models} failed)")
        print("=" * 50)

        # individual model results
        for model_id, result in results.items():
            print(format_model_results(model_id, result))

        print(f"\n⏱️ Total time: {time.time() - start_time:.2f}s")
        print("=" * 50)

    elif "," in args.model:
        # Run specific models in parallel
        model_ids = [model_id.strip() for model_id in args.model.split(",")]

        invalid_models = [model_id for model_id in model_ids if model_id not in MODEL_CONFIGS]
        if invalid_models:
            print(f"Error: The following models are not supported: {', '.join(invalid_models)}")
            print("Use --list to see all supported models.")
            return

        print(f"Processing image with {len(model_ids)} selected models in parallel...")
        results = run_models_in_parallel(args.image, model_ids, args.prompt)

        successful_models = sum(1 for result in results.values() if "error" not in result)
        failed_models = len(results) - successful_models

        print("\n" + "=" * 50)
        print(f"OCR COMPARISON RESULTS ({successful_models} successful, {failed_models} failed)")
        print("=" * 50)

        # individual model results
        for model_id, result in results.items():
            print(format_model_results(model_id, result))

        print(f"\n⏱️ Total time: {time.time() - start_time:.2f}s")
        print("=" * 50)

    else:
        # Run a single model
        if args.model not in MODEL_CONFIGS:
            print(f"Error: Model '{args.model}' not supported.")
            print("Use --list to see all supported models.")
            return

        print(f"\nProcessing with {args.model} model...")
        result = run_ocr_from_ui(args.image, args.model, args.prompt)

        print("\n" + "=" * 50)
        print(f"OCR RESULT FOR {args.model}")
        print("=" * 50)
        print(format_model_results(args.model, result))
        print("=" * 50)


if __name__ == "__main__":
    main()
