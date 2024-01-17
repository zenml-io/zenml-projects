# Write a ZenmL pipeline that writes a pipeline to train iris classifier

from zenml  import pipeline
from zenml.integrations.constants import PYTORCH
from zenml.integrations.pytorch.steps import (
    PyTorchDataLoader,
    PyTorchModelTrainer,
)

    
@step 
def trainer(): 
