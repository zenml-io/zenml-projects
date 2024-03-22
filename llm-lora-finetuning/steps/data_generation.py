from zenml import step

import os
from typing import Any, Dict, List

from argilla import FeedbackDataset
from distilabel.llm import OllamaLLM
from distilabel.pipeline import Pipeline, pipeline
from distilabel.tasks import TextGenerationTask, SelfInstructTask, Prompt
import argilla as rg
from argilla._constants import DEFAULT_API_KEY
from datasets import Dataset
from haystack.nodes import PDFToTextConverter, PreProcessor


@step
def generate_instruction_data(documents: List[Any]) -> FeedbackDataset:
    """Step to generate instruction data."""
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=150,
        split_respect_sentence_boundary=True,
    )
    raw_texts = [{"content": doc.page_content} for doc in documents]
    docs = preprocessor.process(raw_texts)
    inputs = [doc.content for doc in docs]
    instructions_dataset = Dataset.from_dict({"input": inputs})

    instructions_task = SelfInstructTask(
        application_description="An assistant that can answer questions about the open-source MLOps framework ZenML."
    )

    instructions_generator = OllamaLLM(
        model="mixtral",
        task=instructions_task,
    )

    instructions_pipeline = Pipeline(generator=instructions_generator)

    generated_instructions = instructions_pipeline.generate(
        dataset=instructions_dataset, num_generations=1, batch_size=3
    )

    instructions_rag_dataset = generated_instructions.to_argilla()

    # Argilla credentials
    api_url = "https://strickvl-argilla.hf.space"
    api_key = "admin.apikey"

    rg.init(api_url=api_url, api_key=api_key)

    instructions_rag_dataset.push_to_argilla(
        name="ollama_instructions_zenml_rag", workspace="admin"
    )
    return instructions_rag_dataset


@step
def generate_preference_data(instruction_dataset_name: str = None) -> str:
    """Step to generate preference data."""
    return instruction_dataset_name
