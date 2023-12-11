# {% include 'template/license_header' %}

import pandas as pd
import mlflow
from sklearn.base import ClassifierMixin
from zenml import step, log_artifact_metadata
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_evaluator(
    model: ClassifierMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
) -> float:
    """Evaluate a trained model.

    This is an example of a model evaluation step that takes in a model artifact
    previously trained by another step in your pipeline, and a training
    and validation data set pair which it uses to evaluate the model's
    performance. The model metrics are then returned as step output artifacts
    (in this case, the model accuracy on the train and test set).

    The suggested step implementation also outputs some warnings if the model
    performance does not meet some minimum criteria. This is just an example of
    how you can use steps to monitor your model performance and alert you if
    something goes wrong. As an alternative, you can raise an exception in the
    step to force the pipeline run to fail early and all subsequent steps to
    be skipped.

    This step is parameterized to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to use different values for the acceptable model performance thresholds and
    to control whether the pipeline run should fail if the model performance
    does not meet the minimum criteria. See the documentation for more
    information:

        https://docs.zenml.io/user-guide/advanced-guide/configure-steps-pipelines

    Args:
        model: The pre-trained model artifact.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        min_train_accuracy: Minimal acceptable training accuracy value.
        min_test_accuracy: Minimal acceptable testing accuracy value.
        fail_on_accuracy_quality_gates: If `True` a `RuntimeException` is raised
            upon not meeting one of the minimal accuracy thresholds.

    Returns:
        The model accuracy on the test set.

    Raises:
        RuntimeError: if any of accuracies is lower than respective threshold
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    # context = get_step_context()
    # target = context.inputs["dataset_trn"].run_metadata['target'].value
    target = "target"

    # Calculate the model accuracy on the train and test set
    trn_acc = model.score(
        dataset_trn.drop(columns=[target]),
        dataset_trn[target],
    )
    logger.info(f"Train accuracy={trn_acc*100:.2f}%")
    tst_acc = model.score(
        dataset_tst.drop(columns=[target]),
        dataset_tst[target],
    )
    logger.info(f"Test accuracy={tst_acc*100:.2f}%")

    messages = []
    if trn_acc < min_train_accuracy:
        messages.append(
            f"Train accuracy {trn_acc*100:.2f}% is below {min_train_accuracy*100:.2f}% !"
        )
    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test accuracy {tst_acc*100:.2f}% is below {min_test_accuracy*100:.2f}% !"
        )
    else:
        for message in messages:
            logger.warning(message)

    log_artifact_metadata(
        metadata={"train_accuracy": float(trn_acc), "test_accuracy": float(tst_acc)},
        artifact_name="model",
    )

    mlflow.log_metric("train_accuracy", float(trn_acc))
    mlflow.log_metric("test_accuracy",  float(tst_acc))

    return float(trn_acc)
