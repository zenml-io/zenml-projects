#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
import os
import tempfile
from typing import List, Optional
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from steps.get_or_create_dataset import LABELS
from torchvision import models, transforms

from zenml import get_step_context, step
from zenml.client import Client
from zenml.integrations.label_studio.label_studio_utils import (
    get_file_extension,
    is_azure_url,
    is_s3_url,
)
from zenml.io import fileio
from zenml.utils import io_utils

LABEL_MAPPING = {label: idx for idx, label in enumerate(LABELS)}

PIPELINE_NAME = "training_pipeline"
PIPELINE_STEP_NAME = "pytorch_model_trainer"


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=25,
    device="cpu",
):
    """Simplified version of https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html."""
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward; track history only if in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward; optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                dataloaders[phase].dataset
            )
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return model


class CustomDataset:
    """Creates a dataset to be used in the PyTorch model training."""

    def __init__(
        self, image_urls, labels, transforms, artifact_store_path: str
    ) -> None:
        assert len(image_urls) == len(labels)
        self.transforms = transforms

        # Download all images from the artifact store as np.ndarray
        self.images = []
        temp_dir = tempfile.TemporaryDirectory()
        for i, image_url in enumerate(image_urls):
            parts = image_url.split("/")
            if is_s3_url(image_url):
                file_url = f"{artifact_store_path}{urlparse(image_url).path}"
            elif is_azure_url(image_url):
                file_url = "az://" + "/".join(parts[3:])
            else:
                url_path = urlparse(image_url).path
                file_url = f"{artifact_store_path}/{url_path}"
            file_extension = get_file_extension(urlparse(image_url).path)
            path = os.path.join(temp_dir.name, f"{i}{file_extension}")

            io_utils.copy(file_url, path)
            with fileio.open(path, "rb") as f:
                image = np.asarray(Image.open(f))
                self.images.append(image)
        fileio.rmtree(temp_dir.name)

        # Define class-label mapping and map labels
        self.class_label_mapping = LABEL_MAPPING
        self.labels = [self.class_label_mapping[label] for label in labels]

    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        label = self.labels[idx]
        return (image, label)

    def __len__(self):
        return len(self.images)


def _load_last_model() -> Optional[nn.Module]:
    """Return the most recently trained model from this pipeline, or None."""
    last_run = Client().get_pipeline(PIPELINE_NAME).last_successful_run
    return last_run.steps[PIPELINE_STEP_NAME].output.load()


def _is_new_data_available(image_urls: List[str]) -> bool:
    """Find whether new data is available since the last run."""
    # If there are no samples, nothing can be new.
    num_samples = len(image_urls)
    if num_samples == 0:
        return False

    # Else, we check whether we had the same number of samples before.
    last_run = Client().get_pipeline(PIPELINE_NAME).last_successful_run
    last_inputs = last_run.steps[PIPELINE_STEP_NAME].inputs
    last_image_urls = last_inputs["image_urls"].load()
    return len(last_image_urls) != len(image_urls)


def load_pretrained_mobilenetv3(num_classes: int = 2):
    """Load a pretrained mobilenetv3 with fresh classification head."""
    model = models.mobilenet_v3_small(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = nn.Linear(1024, num_classes)
    model.num_classes = num_classes
    return model


def load_mobilenetv3_transforms():
    """Load the transforms required before running mobilenetv3 on data."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    return transforms.Compose([transforms.ToTensor(), weights.transforms()])


@step(enable_cache=False)
def pytorch_model_trainer(
    image_urls: List[str],
    labels: List[str],
    batch_size=1,
    num_epochs=2,
    learning_rate=5e-3,
    device="cpu",
    shuffle=True,
    num_workers=1,
    seed=42,  # don't change: this seed ensures a good train/val split
) -> nn.Module:
    """ZenML step which finetunes or loads a pretrained mobilenetv3 model."""
    # Try to load a model from a previous run, otherwise use a pretrained net
    model = _load_last_model()
    if model is None:
        model = load_pretrained_mobilenetv3()

    # If there is no new data, just return the model
    if not _is_new_data_available(image_urls):
        return model

    context = get_step_context()
    artifact_store_path = context.stack.artifact_store.path

    # Otherwise finetune the model on the current data
    # Write images and labels to torch dataset
    dataset = CustomDataset(
        image_urls=image_urls,
        labels=labels,
        transforms=load_mobilenetv3_transforms(),
        artifact_store_path=artifact_store_path,
    )

    # Split dataset into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    dataset_dict = {
        "train": train_dataset,
        "val": val_dataset,
    }

    # Create training and validation dataloaders
    dataloaders_dict = {
        dataset_type: torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        for dataset_type, dataset in dataset_dict.items()
    }

    # Define optimizer
    optimizer_ft = optim.Adam(
        params=model.classifier[-1].parameters(), lr=learning_rate
    )

    # Define loss
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model = train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer_ft,
        num_epochs=num_epochs,
        device=device,
    )
    return model
