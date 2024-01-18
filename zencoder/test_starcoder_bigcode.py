# Write a zenml pipeline that loads sklearn iris dataset and builds a sklearn classifier 

from  zenml.pipelines import pipeline
from zenml.steps.preprocesser import StandardPreprocesser
from zenml.steps.split import RandomSplit
from zenml.steps.evaluator import TFMAEvaluator
from zenml.steps.trainer import TFFeed
from zenml.steps.deployer import TFServingDeployer
from zenml.steps.preprocesser.standard_preprocesser.standard_preprocesser import \
    StandardPreprocesser
from zenml.steps.split.random_split.random_split import 