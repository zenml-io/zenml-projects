"""Step to download the Enron email dataset from Hugging Face."""

from typing import Annotated

from datasets import Dataset, Features, Sequence, Value, load_dataset
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

EMAIL_DATASET_REPO_ID = "corbt/enron-emails"


@step
def download_enron_data() -> Annotated[Dataset, "raw_emails"]:
    """Download the Enron email dataset from Hugging Face.

    This step fetches the complete Enron email corpus, which contains
    approximately 500,000 emails from Enron employees. The dataset is
    used to create a searchable email database for the agent.

    Returns:
        The raw email dataset from Hugging Face.
    """
    logger.info(f"Downloading email dataset from {EMAIL_DATASET_REPO_ID}...")

    # Define expected schema for type safety
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )

    dataset = load_dataset(
        EMAIL_DATASET_REPO_ID,
        features=expected_features,
        split="train",
    )

    logger.info(f"Downloaded {len(dataset)} emails")
    return dataset
