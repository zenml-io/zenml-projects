from zenml import step, pipeline
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import PIL.Image
from smart_open import open

@dataclass
class SharedConfig:
    """Configuration information shared across project components."""

    # The instance name is the "proper noun" we're teaching the model
    instance_name: str = "blupus cat"
    # That proper noun is usually a member of some class (person, bird),
    # and sharing that information with the model helps it generalize better.
    class_name: str = "ginger cat"  
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-schnell"

@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for plaintext file with urls for images of target instance
    instance_example_urls_file: str = str(
        Path(__file__).parent / "instance_example_urls.txt"
    )

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 3
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117

# load paths to all of the images in a specific directory
def load_image_paths(image_dir: Path) -> List[Path]:
    return list(image_dir.glob("*.png"))

def load_images(image_urls: List[str]) -> Path:
    img_path = Path("./img")

    img_path.mkdir(parents=True, exist_ok=True)
    num_images = 0
    for num_images, url in enumerate(image_urls, start=1):
        with open(url, "rb") as f:
            image = PIL.Image.open(f)
            image.save(img_path / f"{num_images - 1}.png")
    print(f"{num_images} images loaded")

    return img_path


@step
def load_data():
    pass

@step(step_operator="k8s_step_operator")
def train_model():
    pass

@step(step_operator="k8s_step_operator")
def batch_inference():
    pass


@pipeline
def dreambooth_pipeline():
    data = load_data()
    train_model(data)
    batch_inference()
