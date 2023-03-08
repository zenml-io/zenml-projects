from zenml.integrations.evidently.steps import (
    EvidentlyProfileParameters,
    evidently_profile_step,
)

evidently_drift_detector = evidently_profile_step(
    step_name="drift_detector",
    params=EvidentlyProfileParameters(
        # column_mapping=None,
        profile_sections=["datadrift"],
        verbose_level=1,
    ),
)
