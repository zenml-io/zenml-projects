from zenml.steps import step
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.artifacts import DataArtifact
from zenml.io import fileio

from model_config import DatasetConfig
from typing import List, Type
import os
import pickle
import pandas as pd



class ListOfSentenceMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (List[str], )
    ASSOCIATED_ARTIFACT_TYPES = (DataArtifact, )

    def handle_input(self, data_type: Type[List[str]]) -> List[str]:
        """Read from artifact store"""
        super().handle_input(data_type)
        with fileio.open(os.path.join(self.artifact.uri, 'data.p'), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def handle_return(self, my_obj: List[str]) -> None:
        """Write to artifact store"""
        super().handle_return(my_obj)
        with fileio.open(os.path.join(self.artifact.uri, 'data.p'), 'wb') as f:
            pickle.dump(my_obj, f)

@step
def sentence_importer(config: DatasetConfig) -> pd.DataFrame:
    """
    :param path: path to sentence file
    It reads the list of sentences in the txt file
    :return: list of sentences
    """
    sentences = list()
    with open(config.file_path, 'r') as f:
        for line in f:
            sentences.append(line.rstrip())
    # return sentences
    return pd.DataFrame.from_dict({"sentences": sentences})

