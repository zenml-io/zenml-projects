import pandas as pd
from src.configs import PreTrainingConfigs
from src.data.data_loader import CustomDataLoader
from src.data.data_process import ProcessData
from src.data.prepare_data import PrepareDataFrame
from torch.utils.data import DataLoader
from zenml.logger import get_logger
from zenml.steps import Output, step

logger = get_logger(__name__)


@step
def prepare_data() -> Output(processed_dataframe=pd.DataFrame):
    """
    TODO:
    """
    try:
        prep_df = PrepareDataFrame("./data/archive/train.csv")
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
    TODO:
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
    TODO:
    """
    try:
        process_data = ProcessData()
        data_transforms = process_data.augment_data(config)
        return data_transforms
    except Exception as e:
        logger.error(e)
        raise e


@step
def prepare_dataloaders(
    config: PreTrainingConfigs,
) -> Output(train_loader=DataLoader, valid_loader=DataLoader):
    """
    TODO:
    """
    try:
        custom_data_loader = CustomDataLoader(config.n_fold, config)
        train_loader, valid_loader = custom_data_loader.apply_loaders()
        return train_loader, valid_loader
    except Exception as e:
        logger.error(e)
        raise e
