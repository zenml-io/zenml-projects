from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps import (
    EvidentlyReportParameters,
    evidently_report_step,
)

evidently_drift_detector = evidently_report_step(
    step_name="drift_detector",
    params=EvidentlyReportParameters(
        # column_mapping=None,
        metrics=[EvidentlyMetricConfig.metric("DatasetDriftMetric")],
    ),
)
