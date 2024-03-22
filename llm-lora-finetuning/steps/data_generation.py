from zenml import step


@step
def generate_instruction_data() -> None:
    """Step to generate instruction data."""
    pass


@step
def generate_preference_data(instruction_dataset_name: str) -> None:
    """Step to generate preference data."""
    pass
