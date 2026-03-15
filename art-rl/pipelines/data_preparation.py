"""Data preparation pipeline for the email search agent.

This pipeline downloads and prepares all data artifacts needed for training:
1. Downloads the Enron email dataset from Hugging Face
2. Creates a SQLite database with FTS5 for fast email search
3. Loads Q&A scenarios for training and testing

Run this pipeline once before training - artifacts are cached and reused.
"""

from steps.data import (
    create_database,
    download_enron_data,
    load_scenarios,
)
from zenml import Model, pipeline


@pipeline(
    model=Model(
        name="art-email-agent",
        description="Email search agent trained with ART + LangGraph",
        tags=["art", "langgraph", "email-agent", "rl"],
    ),
)
def data_preparation_pipeline(
    db_path: str = "./enron_emails.db",
    max_body_length: int = 5000,
    max_recipients: int = 30,
    train_limit: int = 50,
    test_limit: int = 20,
    max_messages: int = 1,
    seed: int = 42,
):
    """Prepare data artifacts for the email search agent.

    This pipeline is designed to run once and cache all artifacts.
    Subsequent training runs will reuse these cached artifacts.

    Args:
        db_path: Path for the SQLite email database.
        max_body_length: Filter out emails longer than this.
        max_recipients: Filter out emails with more recipients.
        train_limit: Maximum training scenarios to load.
        test_limit: Maximum test scenarios to load.
        max_messages: Filter to scenarios with at most this many source emails.
        seed: Random seed for reproducible scenario shuffling.
    """
    # Step 1: Download raw emails from Hugging Face
    raw_emails = download_enron_data()

    # Step 2: Create searchable SQLite database
    db_path_out = create_database(
        raw_emails=raw_emails,
        db_path=db_path,
        max_body_length=max_body_length,
        max_recipients=max_recipients,
    )

    # Step 3: Load training and test scenarios
    train_scenarios, test_scenarios = load_scenarios(
        train_limit=train_limit,
        test_limit=test_limit,
        max_messages=max_messages,
        seed=seed,
    )

    return db_path_out, train_scenarios, test_scenarios
