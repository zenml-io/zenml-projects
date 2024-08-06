from datasets import Dataset, DatasetDict
from zenml import step


@step
def push_to_hf(train_dataset: Dataset, test_dataset: Dataset, dataset_name: str):
    combined_dataset = DatasetDict(
        {"train": train_dataset, "test": test_dataset}
    )
    combined_dataset.push_to_hub(dataset_name)
