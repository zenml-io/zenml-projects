#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

"""Helper functions for experiment tracking.

This module contains helper functions that allow experiment tracking to be
performed seamlessly across different experiment trackers and stacks.
"""


from typing import Optional
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import (
    MLFlowExperimentTracker,
)

LOCAL_MLFLOW_UI_PORT = 8185


def get_tracker_name() -> Optional[str]:
    """Get the name of the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    return experiment_tracker.name if experiment_tracker else None


def enable_autolog() -> None:
    """Automatically log to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.autolog()


def log_metric(key: str, value: float) -> None:
    """Log a metric to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.log_metric(key, value)


def log_text(text: str, filename: str) -> None:
    """Log a file to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.log_text(text, filename)

def log_model(model, model_name: str) -> None:
    """Log a model to the active experiment tracker."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):
        import mlflow

        mlflow.sklearn.log_model(model, model_name)

def get_current_tracker_run_url() -> Optional[str]:
    """Get the URL of the current experiment tracker run."""

    experiment_tracker = Client().active_stack.experiment_tracker
    if isinstance(experiment_tracker, MLFlowExperimentTracker):

        from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
        import mlflow

        # remote tracking
        if experiment_tracker.config.tracking_uri:

            url = get_tracking_uri()
        else:

            url = f"http://localhost:{LOCAL_MLFLOW_UI_PORT}"

        return f"{url}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}"

    return None
