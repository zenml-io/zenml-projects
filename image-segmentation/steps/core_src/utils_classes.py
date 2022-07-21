import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


class MassUtility:
    """
    TODO: Add docs and fix some error in this.
    """

    def __init__(self, df) -> None:
        self.df = df

    def id2mask(id_):
        idf = self.df[self.df["id"] == id_]
        wh = idf[["height", "width"]].iloc[0]
        shape = (wh.height, wh.width, 3)
        mask = np.zeros(shape, dtype=np.uint8)
        for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
            cdf = idf[idf["class"] == class_]
            rle = cdf.segmentation.squeeze()
            if len(cdf) and not pd.isna(rle):
                mask[..., i] = rle_decode(rle, shape[:2])
        return mask

    def rgb2gray(self, mask):
        pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
        gray_mask = pad_mask.argmax(-1)
        return gray_mask

    def gray2rgb(self, mask):
        rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
        return rgb_mask[..., 1:].astype(mask.dtype)


class ImageUtils:
    def __init__(self) -> None:
        pass

    def load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
        img = img.astype("float32")  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
        return img

    def load_msk(self, path):
        msk = np.load(path)
        msk = msk.astype("float32")
        msk /= 255.0
        return msk

    def show_img(self, img, mask=None):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        plt.imshow(img, cmap="bone")

        if mask is not None:
            plt.imshow(mask, alpha=0.5)
            handles = [
                Rectangle((0, 0), 1, 1, color=_c)
                for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
            ]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(handles, labels)
        plt.axis("off")


class RLEUtils:
    def __init__(self) -> None:
        pass

    def rle_decode(mask_rle, shape):
        """
        mask_rle: run-length as string formatted (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background

        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)  # Needed to align to RLE direction

    # ref.: https://www.kaggle.com/stainsby/fast-tested-rle
    def rle_encode(img):
        """
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formatted
        """
        pixels = img.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(x) for x in runs)
