# Write a iris zenml pipeline

from zenml  import pipeline
from zenml import step

@step
def trainer():
    # 100 