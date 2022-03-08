from typing import Any, Type, Union, List
import pickle
import os

DEFAULT_FILENAME = "PyEnvironment"
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from DQN.model import Agent


class AgentMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = [Agent]

    def handle_input(self, data_type: Type[Any]) -> Union[Agent, Agent]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(self, clf: Union[Agent, Agent],) -> None:
        """Creates a pickle for a sklearn label encoder.
        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)
