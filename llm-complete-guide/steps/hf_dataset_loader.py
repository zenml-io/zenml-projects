from typing import Annotated, Tuple

from datasets import Dataset, load_dataset
from zenml import step
from zenml.integrations.huggingface.materializers.huggingface_datasets_materializer import (
    HFDatasetMaterializer,
)


@step(output_materializers=HFDatasetMaterializer)
def load_hf_dataset(dataset_nama: str) -> (
    Tuple[Annotated[Dataset, "train"], Annotated[Dataset, "test"]]
):
    train_dataset = load_dataset(
        dataset_nama,
        split="train"
    )
    test_dataset = load_dataset(
        dataset_nama,
        split="test"
    )
    return train_dataset, test_dataset


load_hf_dataset()
