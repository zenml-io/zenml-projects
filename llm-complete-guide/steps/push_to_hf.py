from constants import ARGILLA_DATASET_NAME
from datasets import Dataset, DatasetDict
from zenml import step


@step
def push_to_hf(train_dataset: Dataset, test_dataset: Dataset):
    combined_dataset = DatasetDict(
        {"train": train_dataset, "test": test_dataset}
    )
    combined_dataset.push_to_hub(f"zenml/{ARGILLA_DATASET_NAME}")
