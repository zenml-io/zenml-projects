# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import partial
from pathlib import Path

from materializers.directory_materializer import DirectoryMaterializer
from typing_extensions import Annotated
from utils.tokenizer import generate_and_tokenize_prompt, load_tokenizer
from zenml import log_model_metadata, step
from zenml.materializers import BuiltInMaterializer
from zenml.utils.cuda_utils import cleanup_gpu_memory


@step(output_materializers=[DirectoryMaterializer, BuiltInMaterializer])
def prepare_data(
    base_model_id: str,
    system_prompt: str,
    dataset_name: str = "gem/viggo",
    use_fast: bool = True,
    max_train_samples: int = None,
    max_val_samples: int = None,
    max_test_samples: int = None,
) -> Annotated[Path, "datasets_dir"]:
    """Prepare the datasets for finetuning.

    Args:
        base_model_id: The base model id to use.
        system_prompt: The system prompt to use.
        dataset_name: The name of the dataset to use.
        use_fast: Whether to use the fast tokenizer.
        max_train_samples: Maximum number of training samples to use (for CPU or testing).
        max_val_samples: Maximum number of validation samples to use (for CPU or testing).
        max_test_samples: Maximum number of test samples to use (for CPU or testing).

    Returns:
        The path to the datasets directory.
    """
    from datasets import load_dataset
    import logging

    logger = logging.getLogger(__name__)
    cleanup_gpu_memory(force=True)

    # Set default values if None (to prevent validation errors)
    max_train_samples = max_train_samples if max_train_samples is not None else 0
    max_val_samples = max_val_samples if max_val_samples is not None else 0
    max_test_samples = max_test_samples if max_test_samples is not None else 0

    log_model_metadata(
        {
            "system_prompt": system_prompt,
            "base_model_id": base_model_id,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples, 
            "max_test_samples": max_test_samples,
        }
    )

    tokenizer = load_tokenizer(base_model_id, False, use_fast)
    gen_and_tokenize = partial(
        generate_and_tokenize_prompt,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )

    # Load and potentially limit the training dataset
    train_dataset = load_dataset(
        dataset_name,
        split="train",
        trust_remote_code=True,
    )
    if max_train_samples > 0 and max_train_samples < len(train_dataset):
        logger.info(f"Limiting training dataset to {max_train_samples} samples (from {len(train_dataset)})")
        train_dataset = train_dataset.select(range(max_train_samples))
    
    tokenized_train_dataset = train_dataset.map(gen_and_tokenize)
    
    # Load and potentially limit the validation dataset
    eval_dataset = load_dataset(
        dataset_name,
        split="validation",
        trust_remote_code=True,
    )
    if max_val_samples > 0 and max_val_samples < len(eval_dataset):
        logger.info(f"Limiting validation dataset to {max_val_samples} samples (from {len(eval_dataset)})")
        eval_dataset = eval_dataset.select(range(max_val_samples))
        
    tokenized_val_dataset = eval_dataset.map(gen_and_tokenize)
    
    # Load and potentially limit the test dataset
    test_dataset = load_dataset(
        dataset_name,
        split="test",
        trust_remote_code=True,
    )
    if max_test_samples > 0 and max_test_samples < len(test_dataset):
        logger.info(f"Limiting test dataset to {max_test_samples} samples (from {len(test_dataset)})")
        test_dataset = test_dataset.select(range(max_test_samples))

    datasets_path = Path("datasets")
    tokenized_train_dataset.save_to_disk(
        str((datasets_path / "train").absolute())
    )
    tokenized_val_dataset.save_to_disk(str((datasets_path / "val").absolute()))
    test_dataset.save_to_disk(str((datasets_path / "test_raw").absolute()))

    return datasets_path
