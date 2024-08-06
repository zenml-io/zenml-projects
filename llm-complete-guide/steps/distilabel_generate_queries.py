import os
from typing import Annotated, Tuple

from datasets import Dataset
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import GenerateSentencePair
from zenml import step

synthetic_generation_context = """
The text is a chunk from technical documentation of ZenML.
ZenML is an MLOps + LLMOps framework that makes your infrastructure and workflow metadata accessible to data science teams.
Along with prose explanations, the text chunk may include code snippets and logs but these are identifiable from the surrounding backticks.
"""


@step
def generate_synthetic_queries(
    train_dataset: Dataset, test_dataset: Dataset, dataset_name: str, model: str, generation_kwargs: dict
) -> Tuple[
    Annotated[Dataset, "train_with_queries"],
    Annotated[Dataset, "test_with_queries"],
]:
    llm = OpenAILLM(model=model, api_key=os.getenv("OPENAI_API_KEY"))

    with Pipeline(name="generate_embedding_queries") as pipeline:
        load_dataset = LoadDataFromHub(
            # num_examples=20,  # use this for demo purposesc
            output_mappings={"page_content": "anchor"},
        )
        generate_sentence_pair = GenerateSentencePair(
            triplet=True,  # `False` to generate only positive
            action="query",
            llm=llm,
            input_batch_size=10,
            context=synthetic_generation_context,
        )

        load_dataset >> generate_sentence_pair

    train_distiset = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": dataset_name,
                "split": "train",
            },
            generate_sentence_pair.name: {
                "llm": {
                    "generation_kwargs": generation_kwargs
                }
            },
        },
        # use_cache=False, # comment out for demo
    )

    test_distiset = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": dataset_name,
                "split": "test",
            },
            generate_sentence_pair.name: {
                "llm": {
                    "generation_kwargs": generation_kwargs
                }
            },
        },
        # use_cache=False, # comment out for demo
    )

    train_dataset = train_distiset["default"]["train"]
    test_dataset = test_distiset["default"]["train"]

    return train_dataset, test_dataset
