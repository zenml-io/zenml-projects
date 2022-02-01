import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from datetime import date, timedelta
from zenml.steps.step_output import Output
from zenml.steps import step
from zenml.steps.base_step_config import BaseStepConfig


class SklearnSplitterConfig(BaseStepConfig):
    """Config class for the sklearn splitter.
    
    Attributes:
        ratios: Dicts representing split ratios 
        (e.g `{"eval": 0.3, "train": 0.5, "test": 0.2`})
        target_col: The name of the target column.
        shuffle: Whether to shuffle or not.
    """

    ratios: Dict[str, float]
    target_col: str = "FG3M"
    shuffle: bool = True


def split(dataset: pd.DataFrame, target_col: str):
    """Splits data into X and y matrices.

    Args:
        dataset: Full dataset with all the data.
        target_col: Name of column holding target variable.

    Returns:
        Dataframe X with features and Series y with target variable data.
    """
    feature_cols = [x for x in dataset.columns if x != target_col]

    return dataset[feature_cols], dataset[[target_col]]


@step(enable_cache=False)
def sklearn_splitter(
    dataset: pd.DataFrame, config: SklearnSplitterConfig
) -> Output(
    train_x=pd.DataFrame,
    train_y=pd.DataFrame,
    test_x=pd.DataFrame,
    test_y=pd.DataFrame,
    validation_x=pd.DataFrame,
    validation_y=pd.DataFrame,
):
    """Method which is responsible for the splitting logic

    Args:
        dataset: a pandas Dataframe which entire dataset
        config: the configuration for the step

    Returns:
        three dataframes representing the splits
    """

    if config.shuffle:
        dataset = dataset.sample(frac=1)

    dataset_x, dataset_y = split(dataset, config.target_col)

    if (
        any(
            [
                split not in config.ratios
                for split in ["train", "test", "validation"]
            ]
        )
        or len(config.ratios) != 3
    ):
        raise KeyError(
            f"Make sure that you only use 'train', 'test' and "
            f"'validation' as keys in the ratios dict. Current keys: "
            f"{config.ratios.keys()}"
        )

    if sum(config.ratios.values()) != 1:
        raise ValueError(
            f"Make sure that the ratios sum up to 1. Current "
            f"ratios: {config.ratios}"
        )

    train_df_x, test_df_x, train_df_y, test_df_y = train_test_split(
        dataset_x, dataset_y, test_size=config.ratios["test"]
    )

    train_df_x, eval_df_x, train_df_y, eval_df_y = train_test_split(
        train_df_x,
        train_df_y,
        test_size=(
            config.ratios["validation"]
            / (config.ratios["validation"] + config.ratios["train"])
        ),
    )

    return (train_df_x, train_df_y, test_df_x, test_df_y, eval_df_x, eval_df_y)


class SplitConfig(BaseStepConfig):
    date_split: str  # date to split on
    columns: List = []  # optional list of column names to use, empty means all


@step
def date_based_splitter(
    dataset: pd.DataFrame, config: SplitConfig
) -> Output(before=pd.DataFrame, after=pd.DataFrame):
    """Splits data for drift detection."""
    cols = config.columns if config.columns else dataset.columns
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    dataset.set_index("GAME_DATE")
    return (
        dataset[dataset["GAME_DATE"] < config.date_split][cols],
        dataset[dataset["GAME_DATE"] >= config.date_split][cols],
    )


class TrainingSplitConfig(BaseStepConfig):
    """Split config for reference_data_splitter.
    
    Attributes:
        new_data_split_date: Date to split on.
        start_reference_time_frame: Reference time to start from.
        end_reference_time_frame: Reference time to end on.
        columns: optional list of column names to use, empty means all.
    """
    new_data_split_date: str
    start_reference_time_frame: str
    end_reference_time_frame: str
    columns: List = []


@step
def reference_data_splitter(
    dataset: pd.DataFrame, config: TrainingSplitConfig
) -> Output(before=pd.DataFrame, after=pd.DataFrame):
    """Splits data for drift detection."""
    cols = config.columns if config.columns else dataset.columns
    dataset["GAME_DATE"] = pd.to_datetime(dataset["GAME_DATE"])
    dataset.set_index("GAME_DATE")

    reference_dataset = dataset.loc[
        dataset["GAME_DATE"].between(
            config.start_reference_time_frame,
            config.end_reference_time_frame,
            inclusive=True,
        )
    ][cols]

    print(reference_dataset.shape[0])

    new_data = dataset[dataset["GAME_DATE"] >= config.new_data_split_date][
        cols
    ]

    return reference_dataset, new_data


class TimeWindowConfig(BaseStepConfig):
    """Config class for data splitter for the upcoming time window.
    
    Attributes:
        time_window: Number of days to calculate the window with.
    """
    time_window: int


@step
def get_coming_week_data(
    dataset: pd.DataFrame, config: TimeWindowConfig
) -> pd.DataFrame:
    """Get coming week's NBA data.

    Args:
        dataset: DataFrame with all NBA data.
        config: Runtime parameters for the step.

    Returns:
        pd.DataFrame: [description]
    """
    today = date.today().strftime("%Y-%m-%d")
    next_week = (date.today() + timedelta(days=config.time_window)).strftime(
        "%Y-%m-%d"
    )

    format = "%Y-%m-%d %H:%M:%S"
    dataset["DATETIME"] = pd.to_datetime(
        dataset["GAME_DAY"] + " " + dataset["GAME_TIME"] + ":00", format=format
    )
    dataset = dataset.set_index(pd.DatetimeIndex(dataset["DATETIME"]))

    return dataset[
        (dataset["GAME_DAY"] > today) & (dataset["GAME_DAY"] < next_week)
    ]
