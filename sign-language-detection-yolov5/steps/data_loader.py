from genericpath import isdir
import os
from typing import Any, Dict
import numpy as np
from roboflow import Roboflow
from zenml.steps import step, BaseParameters, Output
import cv2
from materializer.dataset_materializer import DatasetMaterializer

class TrainerParameters(BaseParameters):
    """Trainer params"""

    api_key: str = "YOUR_API_KEY"
    workspace: str = "WORKSPACE"
    project: str = "american-sign-language-letters"
    annotation_type: str = "yolov5"


def roboflow_download(api_key:str, workspace:str, project:str, annotation_type:str) -> Any:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(6).download(annotation_type)
    return dataset.location

@step(output_materializers={"train_images": DatasetMaterializer, "val_images": DatasetMaterializer, "test_images": DatasetMaterializer})
def data_loader(
    params: TrainerParameters,
) ->  Output(train_images=Dict,val_images=Dict,test_images=Dict):
    """Loads data from Roboflow"""
    images : dict(str,list(np.ndarray,list)) = {}
    train_images : dict(str,list(np.ndarray,list)) = {}
    valid_images : dict(str,list(np.ndarray,list)) = {}
    test_images : dict(str,list(np.ndarray,list)) = {}
    dataset_path = roboflow_download(params.api_key,params.workspace,params.project,params.annotation_type)
    for folder in os.listdir(dataset_path): 
        if isdir(os.path.join(dataset_path,folder)):
            folder_path = os.path.join(dataset_path, folder)
            for filename in os.listdir(os.path.join(folder_path,"images")):
                img_array = cv2.imread(os.path.join(folder_path,"images",filename))
                load_bboxes = np.genfromtxt(os.path.join(folder_path,"labels",f'{filename[:-4]}.txt'))
                load_bboxes = list(load_bboxes)
                images[os.path.join(folder,filename)] = [img_array,load_bboxes]

            # save each of the sets into different dictionary
            if folder == "train":
                train_images = images
            elif folder == "valid":
                valid_images = images
            else:   
                test_images = images
            images = {}
    return train_images,valid_images,test_images

