import pandas as pd
from torch.utils.data import DataLoader
from zenml.logger import get_logger
from zenml.steps import Output, step

from .core_src.configs import PreTrainingConfigs
from .core_src.data.data_loader import CustomDataLoader
from .core_src.data.data_process import ProcessData
from .core_src.data.prepare_data import PrepareDataFrame


logger = get_logger(__name__)


@step
def prepare_df() -> Output(processed_dataframe=pd.DataFrame):
    """
    It processes and manipulates the masked df.
    """
    try:
        prep_df = PrepareDataFrame("./data/archive/updated_files.csv")
        processed_dataframe = prep_df.prepare_data()
        return processed_dataframe
    except Exception as e:
        logger.error(e)
        raise e


@step
def create_stratified_fold(
    df: pd.DataFrame, config: PreTrainingConfigs
) -> Output(fold_dfs=pd.DataFrame):
    """
    It's a cross validation technique which returns stratified folds from your df.

    Args:
        df: DataFrame.
        config: Pre training config file.
    """
    try:
        process_data = ProcessData()
        fold_dfs = process_data.create_folds(df, config)
        return fold_dfs
    except Exception as e:
        logger.error(e)
        raise e


@step
def apply_augmentations(config: PreTrainingConfigs) -> Output(data_transforms=dict):
    """
    It takes in a config object and returns a dictionary of data transforms.

    Args:
        config: PreTrainingConfigs
    """
    try:
        process_data = ProcessData()
        data_transforms = process_data.augment_data(config)
        return data_transforms
    except Exception as e:
        logger.error(e)
        raise e


def prepare_dataloaders(
    df: pd.DataFrame,
    data_transforms: dict,
    config: PreTrainingConfigs,
) -> Output(train_loader=DataLoader, valid_loader=DataLoader):
    """
    This step takes a dataframe, a dictionary of data transforms, and a config object, and
    returns a train and validation dataloader.

    Args:
        df: pd.DataFrame
        data_transforms: dict
        config: PreTrainingConfigs
    """
    try:
        custom_data_loader = CustomDataLoader(config.n_fold, df, data_transforms, config)
        train_loader, valid_loader = custom_data_loader.apply_loaders()
        return train_loader, valid_loader
    except Exception as e:
        logger.error(e)
        raise e
