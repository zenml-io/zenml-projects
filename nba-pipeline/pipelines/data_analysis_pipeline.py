from zenml.config import DockerSettings
from zenml.integrations.constants import EVIDENTLY, SKLEARN
from zenml.pipelines import pipeline

CURRY_FROM_DOWNTOWN = "2016-02-27"

docker_settings = DockerSettings(required_integrations=[EVIDENTLY, SKLEARN])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def data_analysis_pipeline(importer, drift_splitter, drift_detector, drift_analyzer):
    """Defines a one-time pipeline to detect drift before and after Curry shot.

    Args:
        importer: Import step to query data.
        drift_splitter: Split data step for drift calculation.
        drift_detector: Detect drift step.
        drift_analyzer: Analyze step to parse drift report.
    """
    raw_data = importer()
    reference_dataset, comparison_dataset = drift_splitter(raw_data)
    drift_report, _ = drift_detector(reference_dataset, comparison_dataset)
    drift_analyzer(drift_report)
