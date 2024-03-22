from typing import Any, List

import argilla as rg
from datasets import Dataset
from distilabel.llm import OllamaLLM
from distilabel.pipeline import Pipeline, pipeline
from distilabel.tasks import (
    SelfInstructTask,
    TextGenerationTask,
    UltraFeedbackTask,
)
from haystack.nodes import PreProcessor
from zenml import step

INSTRUCTION_DATASET_NAME = "ollama_instructions_zenml_rag_TEST"
PREFERENCE_DATASET_NAME = "ollama_preferences_zenml_rag_TEST"


@step
def generate_instruction_data(documents: List[Any]) -> str:
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
        name=INSTRUCTION_DATASET_NAME, workspace="admin"
    )
    return INSTRUCTION_DATASET_NAME


@step
def generate_preference_data(
    instruction_dataset_name: str = None,
) -> None:
    """Step to generate preference data."""
    preference_pipeline = pipeline(
        "preference",
        "instruction-following",
        generator=OllamaLLM(
            model="mixtral",
            task=TextGenerationTask(),
            max_new_tokens=256,
            num_threads=2,
            temperature=0.3,
        ),
        labeller=OllamaLLM(
            model="mixtral",
            task=UltraFeedbackTask.for_instruction_following(),
            max_new_tokens=256,
            num_threads=2,
            temperature=0.3,
        ),
        max_new_tokens=256,
        num_threads=2,
        # api_key=os.getenv("OPENAI_API_KEY", None),
        temperature=0.0,
    )

    remote_dataset = rg.FeedbackDataset.from_argilla(
        instruction_dataset_name, workspace="admin"
    )
    instructions_dataset = remote_dataset.pull()
    instructions_dataset = instructions_dataset.format_as("datasets")

    breakpoint()

    instructions_dataset = instructions_dataset.rename_columns(
        {"input": "context", "instructions": "input"}
    )

    preference_dataset = preference_pipeline.generate(
        instructions_dataset,  # type: ignore
        num_generations=2,
        batch_size=8,
        display_progress_bar=True,
    )

    preference_rg_dataset = preference_dataset.to_argilla()

    # Adding the context as a metadata property in the new Feedback dataset, as this
    # information will be useful later.
    for record_feedback, record_huggingface in zip(
        preference_rg_dataset, preference_dataset
    ):
        record_feedback.metadata["context"] = record_huggingface["context"]

    preference_rg_dataset.push_to_argilla(
        name=PREFERENCE_DATASET_NAME, workspace="admin"
    )

    preference_rg_dataset.push_to_huggingface(
        f"strickvl/{PREFERENCE_DATASET_NAME}"
    )
