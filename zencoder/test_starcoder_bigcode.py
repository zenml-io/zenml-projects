# Write a zenml pipeline that loads sklearn iris dataset and builds a sklearn classifier 

from zenml.pipelines import pipeline
from zenml.steps.preprocesser import StandardPreprocesser
from zenml.steps.split import RandomSplit
from zenml.steps.evaluator import TFMAEvaluator
from zenml.steps.trainer import TFFeed
from zenml.steps.deployer import TFServingDeployer
from zenml.steps.preprocesser.standard_preprocesser.standard_preprocesser import \
    StandardPreprocesser

@pipeline
def tf_mnist_pipeline(epochs: int = 5, lr: float = 0.001):
    """Links all the steps together in a pipeline."""
    # Link all the steps together by calling them and passing the output
    # of one step as the input

#     x_train, x_test, y_train, y_test = RandomSplit(test_size=0.2)(
#         dataset=iris_data_loader()
#     )
    x_train, x_test, y_train, y_test = StandardPreprocesser(
        test_size=0.2,
        random_state=42,
    )(
        dataset=iris_data_loader()
    )
    model = TFFeed(epochs=epochs, lr=lr)(
        x_train=x_train

        
