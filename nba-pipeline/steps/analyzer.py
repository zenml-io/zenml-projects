import json

from zenml.steps import step


@step
def analyze_drift(
    datadrift: str,
) -> bool:
    """Analyze the Evidently drift report and return a true/false value indicating
    whether data drift was detected.

    Args:
        datadrift: datadrift dictionary created by evidently
    """
    drift = json.loads(datadrift)["metrics"][0]["result"]["dataset_drift"]

    print("Drift detected" if drift else "No drift detected")
    return drift
