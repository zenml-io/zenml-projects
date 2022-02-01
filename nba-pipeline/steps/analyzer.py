from zenml.steps import step


@step
def analyze_drift(
    datadrift: dict,
) -> bool:
    """Analyze the Evidently drift report and return a true/false value indicating
    whether data drift was detected.

    Args:
        datadrift: datadrift dictionary created by evidently
    """
    drift = datadrift["data_drift"]["data"]["metrics"]["dataset_drift"]
    print("Drift detected" if drift else "No drift detected")
    return drift
