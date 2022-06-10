import pandas as pd
from zenml.logger import get_logger

logger = get_logger(__name__)


class PrepareDataFrame:
    """Prepare data for data loaders."""

    def __init__(self, df_path: str):
        """Initialize the class."""
        self.df_path = df_path

    def prepare_data(self) -> pd.DataFrame:
        """Prepare data for data loaders."""
        try:
            df = pd.read_csv(self.df_path)
            df["segmentation"] = df.segmentation.fillna("")
            df["rle_len"] = df.segmentation.map(len)
            df["mask_path"] = df.mask_path.str.replace("/png/", "/np").str.replace(".png", ".npy")

            df2 = df.groupby(["id"])["segmentation"].agg(list).to_frame().reset_index()
            df2 = df2.merge(df.groupby(["id"])["rle_len"].agg(sum).to_frame().reset_index())

            df = df.drop(columns=["segmentation", "class", "rle_len"])
            df = df.groupby(["id"]).head(1).reset_index(drop=True)
            df = df.merge(df2, on=["id"])
            df["empty"] = df.rle_len == 0
            logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            logger.error(e)
            raise e
