import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from urllib3.exceptions import NewConnectionError
from zenml.steps import BaseParameters, step


class ImporterConfig(BaseParameters):
    """Parameters for the `importer` step.

    Attributes:
        seasons: List of seasons to query NBA API historically.
    """

    seasons = [
        "2000-01",
        "2001-02",
        "2002-03",
        "2003-04",
        "2004-05",
        "2005-06",
        "2006-07",
        "2007-08",
        "2008-09",
        "2009-10",
        "2010-11",
        "2011-12",
        "2012-13",
        "2013-14",
        "2014-15",
        "2015-16",
        "2016-17",
        "2017-18",
        "2018-19",
        "2019-20",
        "2020-21",
        "2021-22",
    ]


@step
def game_data_importer_offline() -> pd.DataFrame:
    """Reads an offline season data downloaded from NBA API and returns a pd.DataFrame. The
    pd.Dataframe contains the following columns:

    |SEASON_ID|...|TEAM_ABBREVIATION|...|GAME_ID|GAME_DATE|...|FG3M|
    """
    print("Using offline data from the NBA API.")
    df = pd.read_csv("season_data.csv")
    return df


@step
def game_data_importer(config: ImporterConfig) -> pd.DataFrame:
    """Downloads season data from NBA API and returns a pd.DataFrame. The
    pd.Dataframe contains the following columns:

    |SEASON_ID|...|TEAM_ABBREVIATION|...|GAME_ID|GAME_DATE|...|FG3M|
    """
    dataframes = []
    for season in config.seasons:
        try:
            print(f"Fetching data for season: {season}")
            dataframes.append(
                leaguegamelog.LeagueGameLog(
                    season=season, timeout=180
                ).get_data_frames()[0]
            )
        except (ConnectionError, NewConnectionError):
            pass
        # sleep so as not to bomb api server :-)
        time.sleep(2)

    return pd.concat(dataframes)


import json
import urllib.request


class SeasonScheduleConfig(BaseParameters):
    """Config for the `import_season_schedule` step.

    Attributes:
        current_season: The current season as a string, e.g. `2021-22`.
    """

    current_season: str


@step
def import_season_schedule_offline() -> pd.DataFrame:
    """Reads an offline season data of the current season (2021-22) downloaded from NBA API and returns a pd.DataFrame."""
    print("Using offline data from the NBA API.")
    df = pd.read_csv("current_season.csv")
    return df


@step(enable_cache=False)
def import_season_schedule(config: SeasonScheduleConfig) -> pd.DataFrame:
    """Imports the current seasons schedule for the NBA API."""
    current_season_schedule_endpoint = f"https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{config.current_season[:-3]}/league/00_full_schedule.json"

    with urllib.request.urlopen(current_season_schedule_endpoint) as url:
        current_season_schedule = json.loads(url.read().decode())
    games = []
    for season in current_season_schedule["lscd"]:
        for game in season["mscd"]["g"]:
            games.append(
                {
                    "SEASON_ID": int("22" + config.current_season[1:4]),
                    "TEAM_ABBREVIATION": game["h"]["ta"],
                    "OPPONENT_TEAM_ABBREVIATION": game["v"]["ta"],
                    "GAME_DAY": game["gdtutc"],
                    "GAME_TIME": game["utctm"],
                }
            )
            games.append(
                {
                    "SEASON_ID": int("22" + config.current_season[1:4]),
                    "TEAM_ABBREVIATION": game["v"]["ta"],
                    "OPPONENT_TEAM_ABBREVIATION": game["h"]["ta"],
                    "GAME_DAY": game["gdtutc"],
                    "GAME_TIME": game["utctm"],
                }
            )

    return pd.DataFrame.from_dict(games)
