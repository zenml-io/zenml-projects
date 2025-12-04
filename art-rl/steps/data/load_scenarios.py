"""Step to load training and test scenarios from Hugging Face."""

from typing import Annotated, List, Tuple

from datasets import load_dataset
from environment.models import Scenario
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

SCENARIO_DATASET_REPO_ID = "corbt/enron_emails_sample_questions"


@step
def load_scenarios(
    train_limit: int = 50,
    test_limit: int = 20,
    max_messages: int = 1,
    seed: int = 42,
) -> Tuple[
    Annotated[List[Scenario], "train_scenarios"],
    Annotated[List[Scenario], "test_scenarios"],
]:
    """Load Q&A scenarios from Hugging Face for training and evaluation.

    Each scenario contains:
    - A question about emails in a user's inbox
    - The reference answer
    - Message IDs of the emails containing the answer
    - Metadata (inbox address, query date, realism score)

    Args:
        train_limit: Maximum number of training scenarios to load.
        test_limit: Maximum number of test scenarios to load.
        max_messages: Filter to scenarios with at most this many source emails.
            Simpler scenarios (fewer source emails) are easier to learn from.
        seed: Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_scenarios, test_scenarios).
    """
    logger.info(f"Loading scenarios from {SCENARIO_DATASET_REPO_ID}...")

    # Load train split
    train_dataset = load_dataset(SCENARIO_DATASET_REPO_ID, split="train")
    test_dataset = load_dataset(SCENARIO_DATASET_REPO_ID, split="test")

    # Filter by max_messages
    if max_messages is not None:
        train_dataset = train_dataset.filter(
            lambda x: len(x["message_ids"]) <= max_messages
        )
        test_dataset = test_dataset.filter(
            lambda x: len(x["message_ids"]) <= max_messages
        )

    # Shuffle with seed for reproducibility
    train_dataset = train_dataset.shuffle(seed=seed)
    test_dataset = test_dataset.shuffle(seed=seed)

    # Convert to Scenario objects
    train_scenarios = [Scenario(**row, split="train") for row in train_dataset]
    test_scenarios = [Scenario(**row, split="test") for row in test_dataset]

    # Apply limits
    if train_limit:
        train_scenarios = train_scenarios[:train_limit]
    if test_limit:
        test_scenarios = test_scenarios[:test_limit]

    logger.info(f"Loaded {len(train_scenarios)} training scenarios")
    logger.info(f"Loaded {len(test_scenarios)} test scenarios")

    # Log sample scenario
    if train_scenarios:
        sample = train_scenarios[0]
        logger.info(f"Sample scenario: {sample.question[:100]}...")

    return train_scenarios, test_scenarios
