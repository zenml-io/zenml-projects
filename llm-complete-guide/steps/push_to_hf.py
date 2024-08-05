from datasets import Dataset
from zenml import step


@step
def push_to_hf(train_dataset: Dataset, test_dataset: Dataset):
    combined_dataset = DatasetDict(
        {"train": train_dataset, "test": test_dataset}
    )
    combined_dataset.push_to_hub(
        "zenml/rag_qa_embedding_questions_0_60_0_distilabel"
    )
