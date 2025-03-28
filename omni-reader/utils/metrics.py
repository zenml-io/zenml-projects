"""This module contains the metrics for the project."""

from jiwer import cer, wer


def compare_results(
    ground_truth: str, gemma_text: str, mistral_text: str
) -> dict:
    """Compares Gemma 3 and Mistral OCR results with the ground truth.

    Args:
        ground_truth (str): The ground truth text.
        gemma_text (str): The text extracted by Gemma 3.
        mistral_text (str): The text extracted by Mistral OCR.

    Returns:
        dict: A dictionary containing the CER and WER for each model.
    """
    metrics = {
        "Gemma CER": cer(ground_truth, gemma_text),
        "Gemma WER": wer(ground_truth, gemma_text),
        "Mistral CER": cer(ground_truth, mistral_text),
        "Mistral WER": wer(ground_truth, mistral_text),
    }

    return metrics
