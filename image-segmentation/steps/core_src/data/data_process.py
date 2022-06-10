import albumentations as A
import cv2
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..configs import PreTrainingConfigs


class ProcessData:
    def __init__(self) -> None:
        pass

    def create_folds(self, df: pd.DataFrame, config: PreTrainingConfigs) -> pd.DataFrame:
        """
        `StratifiedGroupKFold` is a class that splits the data into train and validation sets, and the
        `split` function of this class returns the indices of the train and validation sets.

        df: the dataframe that contains the data
        config: PreTrainingConfigs
        """
        skf = StratifiedGroupKFold(n_splits=config.n_fold, shuffle=True, random_state=config.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["empty"], groups=df["case"])):
            df.loc[val_idx, "fold"] = fold
        return df

    def augment_data(self, config: PreTrainingConfigs) -> dict:
        """
        It takes in a config object and returns a dictionary of data transforms for training and validation

        config: PreTrainingConfigs
        """
        data_transforms = {
            "train": A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
                    A.HorizontalFlip(p=0.5),
                    #         A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5
                    ),
                    A.OneOf(
                        [
                            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                            # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                        ],
                        p=0.25,
                    ),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=config.img_size[0] // 20,
                        max_width=config.img_size[1] // 20,
                        min_holes=5,
                        fill_value=0,
                        mask_fill_value=0,
                        p=0.5,
                    ),
                ],
                p=1.0,
            ),
            "valid": A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_NEAREST),
                ],
                p=1.0,
            ),
        }
        return data_transforms
