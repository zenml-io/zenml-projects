from zenml.pipelines import pipeline
from zenml.integrations.constants import SKLEARN

CURRY_FROM_DOWNTOWN = "2016-02-27"


@pipeline(required_integrations=SKLEARN)
def data_analysis_pipeline(
    importer, drift_splitter, drift_detector, drift_analyzer
):
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


if __name__ == "__main__":
    from steps.importer import game_data_importer
    from steps.splitter import date_based_splitter, SplitConfig
    from steps.analyzer import analyze_drift
    from steps.profiler import evidently_drift_detector

    # Initialize the pipeline
    eda_pipeline = data_analysis_pipeline(
        importer=game_data_importer(),
        drift_splitter=date_based_splitter(
            SplitConfig(date_split=CURRY_FROM_DOWNTOWN, columns=["FG3M"])
        ),
        drift_detector=evidently_drift_detector,
        drift_analyzer=analyze_drift(),
    )

    eda_pipeline.run()
