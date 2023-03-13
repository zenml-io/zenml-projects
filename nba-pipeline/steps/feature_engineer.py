from datetime import date
from typing import List

import pandas as pd
from zenml.steps import BaseParameters, step


class FeatureEngineererConfig(BaseParameters):
    """Config class for the sklearn splitter.

    Attributes:
      history_length: Amount of years of data to look at during training
      select_features: Features to use as input during training
    """

    history_length: int = 2
    select_features: List[str] = ["SEASON_ID", "TEAM_ABBREVIATION", "FG3M"]


def limit_timeframe(
    dataset: pd.DataFrame, years_to_subtract: int
) -> pd.DataFrame:
    """We use only the last couple years of data in order to not fit
    to outdated playing styles"""

    today = date.today()
    oldest_data = today.replace(year=today.year - years_to_subtract)

    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    dataset = dataset.set_index(pd.DatetimeIndex(dataset["GAME_DATE"]))

    dataset = dataset[dataset["GAME_DATE"] > pd.to_datetime(oldest_data)]

    return dataset


@step
def feature_engineer(
    dataset: pd.DataFrame, config: FeatureEngineererConfig
) -> pd.DataFrame:
    """Preprocesses data columns and add opponent to each match.
    Return:
      Returns a dataframe with the following columns:
      |SEASON_ID|TEAM_ABBREVIATION|OPPONENT_TEAM_ABBREVIATION|FG3M|

    """

    def add_opponent(match_rows):
        """Within a given game_id there is two rows, one for each team.
        For each of these rows the corresponding opponent is added to a new
        OPPONENT_TEAM_ABBREVIATION - column
        """
        teams_in_match = pd.unique(match_rows["TEAM_ABBREVIATION"])
        opponent_dir = {
            teams_in_match[i]: teams_in_match[::-1][i] for i in [0, 1]
        }
        match_rows["OPPONENT_TEAM_ABBREVIATION"] = match_rows.apply(
            lambda x: opponent_dir[x["TEAM_ABBREVIATION"]], axis=1
        )
        return match_rows

    select_features = config.select_features + ["GAME_ID"]

    dataset = limit_timeframe(
        dataset=dataset, years_to_subtract=config.history_length
    )

    return (
        dataset[select_features]
        .groupby("GAME_ID")
        .apply(add_opponent)
        .drop(columns=["GAME_ID"])
    )
