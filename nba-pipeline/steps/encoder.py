import os
import pandas as pd
import numpy as np
from typing import Any, Type, Union, List
import pickle
from sklearn import preprocessing

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.artifacts import ModelArtifact
from zenml.steps.step_output import Output
from zenml.io import fileio
from zenml.steps import step, StepContext

from .utils import get_label_encoder, apply_encoder

DEFAULT_FILENAME = "label_encoder"


class SklearnLEMaterializer(BaseMaterializer):
    """Materializer to read data to and from sklearn."""

    ASSOCIATED_TYPES = [
        preprocessing.LabelEncoder,
        preprocessing.OneHotEncoder,
    ]

    ASSOCIATED_ARTIFACT_TYPES = [ModelArtifact]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[preprocessing.LabelEncoder, preprocessing.OneHotEncoder]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
        self,
        clf: Union[preprocessing.LabelEncoder, preprocessing.OneHotEncoder],
    ) -> None:
        """Creates a pickle for a sklearn label encoder.

        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)


@step(enable_cache=False)
def data_encoder(
    pandas_df: pd.DataFrame,
) -> Output(
    encoded_data=pd.DataFrame,
    le_seasons=preprocessing.LabelEncoder,
    ohe_teams=preprocessing.OneHotEncoder,
):
    """Encode columns with label encoder/ one hot encoder.

    Args:
        pandas_df: Pandas df containing at least the following columns
                   SEASON_ID, TEAM_ABBREVIATION, OPPONENT_TEAM_ABBREVIATION
    Returns:
        encoded_data: Dataframe with encoded data
        le_seasons: Label encoder for SEASON_ID
        ohe_teams: One hot encoder for team abbreviations
    """
    # convert categorical to ints
    le_seasons = preprocessing.LabelEncoder()
    le_seasons.fit(pandas_df["SEASON_ID"])

    ohe_teams = preprocessing.OneHotEncoder(
        dtype=np.int32, handle_unknown="ignore"
    )
    ohe_teams.fit(pandas_df["TEAM_ABBREVIATION"].values.reshape(-1, 1))

    new_df = apply_encoder(
        label_encoder=le_seasons,
        one_hot_encoder=ohe_teams,
        dataframe=pandas_df,
    )

    return new_df, le_seasons, ohe_teams


@step
def encode_columns_and_clean(
        context: StepContext,
        pandas_df: pd.DataFrame,
) -> Output(encoded_data=pd.DataFrame, le_season=preprocessing.LabelEncoder):
    """Encode columns with label encoder/ one hot encoder. Remove games that
    do not have a set game date yet.

    Args:
        context: Step context to access previous runs
        pandas_df: Pandas df containing at least the following columns
                   GAME_TIME, SEASON_ID, TEAM_ABBREVIATION,
                   OPPONENT_TEAM_ABBREVIATION
    Returns:
        encoded_data: Dataframe with encoded data
        le_seasons: Label encoder for SEASON_ID
    """
    # convert categorical to ints
    le_seasons = get_label_encoder(name="le_seasons", context=context)

    ohe_teams = get_label_encoder(name="ohe_teams", context=context)

    # Clean data with missing date
    pandas_df = pandas_df.drop(pandas_df[pandas_df["GAME_DAY"] == "TBD"].index)
    pandas_df["GAME_TIME"].mask(
        pandas_df["GAME_TIME"] == "TBD", "00:00:00", inplace=True
    )

    # Apply label encoders using the same function as during training
    new_df = apply_encoder(
        label_encoder=le_seasons,
        one_hot_encoder=ohe_teams,
        dataframe=pandas_df,
    )

    return new_df, le_seasons
