from typing import Any, Dict, List
import albumentations as A
import pandas as pd
import numpy as np
import cv2
import os
import albumentations as A

from zenml.steps import step, BaseParameters, Output
#from zenml.materializers import BuiltInContainerMaterializer


class AugmenterParameters(BaseParameters):
    """Trainer params"""

    set_type = "valid"


 
#@step(output_materializers={"augmented_images": BuiltInContainerMaterializer})
@step
def valid_augmenter(
    #params:AugmenterParameters,
    images: Dict,
) -> Output(augmented_images=Dict):
    """Loads data from Roboflow"""

    #rows = []
    augmented_images : dict(str,list(np.ndarray,list)) = {}
    for key, value  in images.items():
        image = value[0]
        load_bboxes = value[1]
        bbox_cat = int(load_bboxes[0])
        bboxes = load_bboxes[1:]
        img_ht = int(image.shape[0])
        img_wd = int(image.shape[1])
        bb_width = int(round(bboxes[2] * image.shape[1], 0))
        bb_height = int(round(bboxes[3] * image.shape[0], 0))
        x_min = int((img_wd * bboxes[0]) - (bb_width/2))
        x_max = int((img_wd * bboxes[0]) + (bb_width/2))
        y_min = int(img_ht * bboxes[1] - (bb_height/2))
        y_max = int(img_ht * bboxes[1] + (bb_height/2))
        new_bboxes = [x_min, y_min, x_max, y_max]

        # Creating 25 augmented images to compensate for the small dataset
        for i in range(25):
            #class_name = bbox_cat
            augmented = aug(image=image, bboxes=[new_bboxes], class_name=[bbox_cat])
            image_name = f'{key[:-4]}_{i}.jpg'
            #text_name = f'{key[:-4]}_{i}.txt'
            boxes = []
            for bbox in augmented['bboxes']:
                x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)
                # Converting back to Yolo format
                x_center = (x_min + x_max) / 2
                x_center /= 1024
                y_center = (y_min + y_max) / 2
                y_center /= 1024
                w = (x_max - x_min) / 1024
                h = (y_max - y_min) /1024
                new_bbox = [bbox_cat, x_center, y_center, w, h]
                boxes.append(new_bbox)  
                #rows.append({
                #    'image_id': f'{params.augment}/{params.img}/{image_name}',
                #    'bbox': new_bbox
                #})
        if len(boxes) == 0:
            continue
        augmented_images[image_name] = [augmented['image'],boxes]
        #cv2.imwrite(f'{params.augment}/{params.img}/{image_name}', augmented['image'])
        #np.savetxt(
        #        # Outputting .txt file to appropriate train/validation folders
        #        os.path.join(f'{params.augment}/{params.labels}/{text_name}'),
        #        np.array(boxes),
        #        fmt=["%d", "%f", "%f", "%f","%f"]
        #    )
    return augmented_images


bbox_params = A.BboxParams(
    format= 'pascal_voc',
    min_area=1,
    min_visibility=0.5,
    label_fields=['class_name']
)

aug = A.Compose([
    A.LongestMaxSize(max_size=1024),
    A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0,p=1.0),
    A.ShiftScaleRotate(shift_limit=.25, scale_limit=0.2, p=0.3),
    A.RandomSizedCrop((900, 1000), 1024, 1024, p=.2),
    A.HorizontalFlip(p=.5),
    A.Rotate(limit=30,p=.8),
    A.MultiplicativeNoise(p=.2),
    A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=.3),
    A.Blur(blur_limit=25, p=.2),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.35,p=.5),
    A.HueSaturationValue(p=.3),
    #A.CoarseDropout(max_holes=9,min_width=30, max_width=250, min_height=30, max_height=250,p=.2),
    A.OneOf([A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(var_limit=(10, 50),mean=50)], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1)], p=0.2),
    A.OneOf([
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
        A.RandomGamma(gamma_limit=[50,200], p=.2),
        A.ToGray()], p=0.3),
        A.NoOp(p=.04)
]
    , bbox_params=bbox_params)
