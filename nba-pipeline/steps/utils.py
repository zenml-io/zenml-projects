import pandas as pd
from sklearn import preprocessing

from zenml.steps import StepContext
from zenml.post_execution import get_pipeline


def get_label_encoder(
        context: StepContext,
        name: str,
        pipeline_name: str = "training_pipeline",
        step_name: str = "encoder",
) -> preprocessing.LabelEncoder:
    """Returns the label encoder from a given pipeline's latest run.

    Args:
        context: Step context to access previous runs
        name: Output name for the specific encoder
        pipeline_name: Name of the pipeline to retrieve the
                       encoder from
        step_name: Name of the step that returns the encoder

    Return:
        label_encoder: The LabelEncoder used during training
    """
    training_pipeline = get_pipeline(
        pipeline_name=pipeline_name)
    return (
        training_pipeline.runs[-1]
            .get_step(name=step_name)
            .outputs[name]
            .read()
    )


def apply_encoder(
        label_encoder: preprocessing.LabelEncoder,
        one_hot_encoder: preprocessing.OneHotEncoder,
        dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Use Encoders to turn the categorical columns into LabelEncoded/
    OneHotEncoded columns.

    Args:
        label_encoder: Label Encoder used for Season ID
        one_hot_encoder: Used for the teams
        dataframe: Initial dataset

    Returns:
        new_df: With Season ID and Team Abbreviations replaced by encoded 
        columns

    """

    dataframe["SEASON_ID"] = label_encoder.fit_transform(
        dataframe["SEASON_ID"]
    )

    one_hot_team = one_hot_encoder.transform(
        dataframe["TEAM_ABBREVIATION"].values.reshape(-1, 1)
    ).todense()
    one_hot_team_df = pd.DataFrame(
        one_hot_team,
        columns=one_hot_encoder.categories_[0],
        index=dataframe.index,
    )

    one_hot_opponent = one_hot_encoder.transform(
        dataframe["OPPONENT_TEAM_ABBREVIATION"].values.reshape(-1, 1)
    ).todense()
    one_hot_opponent_df = pd.DataFrame(
        one_hot_opponent,
        columns=one_hot_encoder.categories_[0],
        index=dataframe.index,
    )

    dataframe = dataframe.drop(
        ["TEAM_ABBREVIATION", "OPPONENT_TEAM_ABBREVIATION"], axis=1
    )

    new_df = pd.concat(
        [
            dataframe,
            one_hot_team_df.add_prefix("home_"),
            one_hot_opponent_df.add_prefix("away_"),
        ],
        axis=1,
    )

    return new_df
