import os
import random
import shutil
import zipfile
import yaml

def unzip_dataset(zip_path: str, extract_dir: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def split_dataset(data_path: str, ratio: tuple = (0.7, 0.2, 0.1), seed: int = None):
    # Ensure the ratio is correct
    assert sum(ratio) == 1.0

    # Ensure the paths exist
    for subdir in ['images', 'labels']:
        for split in ['train', 'test', 'val']:
            os.makedirs(os.path.join(data_path, subdir, split), exist_ok=True)

    # Get a list of all files
    files = next(os.walk(os.path.join(data_path, 'images')))[2]

    # Seed to get consistent results
    if seed is not None:
        random.seed(seed)

    # Shuffle the list of files
    random.shuffle(files)

    # Calculate the indices to split at
    split_indices = [0] + [round(x * len(files)) for x in ratio] + [None]

    for split, start, end in zip(['train', 'test', 'val'], split_indices[:-1], split_indices[1:]):
        for file in files[start:end]:
            for subdir in ['images', 'labels']:
                # Move each file to the proper split
                shutil.move(os.path.join(data_path, subdir, file),
                            os.path.join(data_path, subdir, split, file))


def generate_yaml(data_path: str, yaml_path: str = None):
    if yaml_path is None:
        yaml_path = data_path
    # Read classes from classes.txt
    with open(os.path.join(data_path, 'classes.txt')) as f:
        classes = [line.strip() for line in f]

    # Define yaml data
    data = {
        'path': os.path.abspath(data_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)}
    }

    # Write to yaml file
    yaml_path = os.path.join(data_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    return yaml_path 
