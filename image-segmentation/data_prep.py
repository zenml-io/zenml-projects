import pandas as pd


def manipulate_process_dataframe():
    train_csv = pd.read_csv(
        "/Users/ayushsingh/Documents/zenfiles/image-segmentation/data/archive/train.csv"
    )
    for i in range(len(train_csv)):
        # change image_path by replacing /kaggle/input/ to /Users/ayushsingh/Documents/zenfiles/image-segmentation/data/
        train_csv["image_path"].iloc[i] = (
            train_csv["image_path"]
            .iloc[i]
            .replace(
                "/kaggle/input/", "/Users/ayushsingh/Documents/zenfiles/image-segmentation/data/"
            )
        )
        train_csv["mask_path"].iloc[i] = (
            train_csv["mask_path"]
            .iloc[i]
            .replace(
                "/kaggle/input/uwmgi-mask-dataset",
                "/Users/ayushsingh/Documents/zenfiles/image-segmentation/data/archive",
            )
        )
    train_csv.to_csv("updated_files.csv", index=False)


if __name__ == "__main__":
    manipulate_process_dataframe()
