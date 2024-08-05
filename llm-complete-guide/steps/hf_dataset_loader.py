from typing import Annotated, Tuple

from datasets import Dataset, load_dataset
from zenml import step
from zenml.integrations.huggingface.materializers.huggingface_datasets_materializer import (
    HFDatasetMaterializer,
)


@step(output_materializers=HFDatasetMaterializer)
def load_hf_dataset() -> (
    Tuple[Annotated[Dataset, "train"], Annotated[Dataset, "test"]]
):
    train_dataset = load_dataset(
        "zenml/rag_qa_embedding_questions_0_60_0_distilabel", split="train"
    )
    test_dataset = load_dataset(
        "zenml/rag_qa_embedding_questions_0_60_0_distilabel", split="test"
    )
    return train_dataset, test_dataset


load_hf_dataset()
