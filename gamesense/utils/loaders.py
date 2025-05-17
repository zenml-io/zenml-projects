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

from pathlib import Path
from typing import Any, Tuple, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

from utils.logging import print_trainable_parameters


def load_base_model(
    base_model_id: str,
    is_training: bool = True,
    use_accelerate: bool = False,
    should_print: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    cpu_only: bool = False,
) -> Union[Any, Tuple[Any, Dataset, Dataset]]:
    """Load the base model.

    Args:
        base_model_id: The base model id to use.
        is_training: Whether the model should be prepared for training or not.
            If True, the Lora parameters will be enabled and PEFT will be
            applied.
        use_accelerate: Whether to use the Accelerate library for training.
        should_print: Whether to print the trainable parameters.
        load_in_8bit: Whether to load the model in 8-bit mode.
        load_in_4bit: Whether to load the model in 4-bit mode.
        cpu_only: Whether to force using CPU only and disable quantization.

    Returns:
        The base model.
    """
    import logging

    from accelerate import Accelerator
    from transformers import BitsAndBytesConfig

    logger = logging.getLogger(__name__)

    # Explicitly disable MPS when in CPU-only mode
    if cpu_only:
        import os

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        # Force PyTorch to not use MPS
        torch._C._set_mps_enabled(False) if hasattr(
            torch._C, "_set_mps_enabled"
        ) else None
        # Set default device to CPU explicitly
        torch.set_default_device("cpu")
        logger.warning("Disabled MPS device for CPU-only mode.")

    if use_accelerate:
        accelerator = Accelerator()
        device_map = {"": accelerator.process_index}
    else:
        # Check for available devices and use the best one
        if cpu_only:
            device_map = {"": "cpu"}
        elif torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        elif torch.backends.mps.is_available() and not cpu_only:
            device_map = {"": "mps"}
        else:
            device_map = {"": "cpu"}

    # Only use BitsAndBytes config if CUDA is available and quantization is requested
    # and we're not in CPU-only mode
    if (
        (load_in_8bit or load_in_4bit)
        and torch.cuda.is_available()
        and not cpu_only
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None
        # Reset these flags if CUDA is not available or in CPU-only mode
        load_in_8bit = False
        load_in_4bit = False

    # Print device information for debugging
    if should_print:
        print(f"Loading model on device: {device_map}")

    # Use half precision for CPU to reduce memory usage if not in training
    torch_dtype = (
        torch.float16 if device_map[""] == "cpu" and not is_training else None
    )

    # Check if it's a Phi model
    is_phi_model = "phi" in base_model_id.lower()

    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        # Use low_cpu_mem_usage for CPU training to minimize memory usage
        "low_cpu_mem_usage": device_map[""] == "cpu",
    }

    # Add special config for Phi models on CPU to avoid cache issues
    if is_phi_model and (cpu_only or device_map[""] == "cpu"):
        if should_print:
            print(
                "Loading Phi model on CPU with special configuration to avoid caching issues"
            )
        model_kwargs["use_flash_attention_2"] = False
        # Set attn_implementation to eager for Phi models on CPU
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    # For Phi models on CPU, disable kv cache feature to avoid errors
    if is_phi_model and (cpu_only or device_map[""] == "cpu"):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            if should_print:
                print("Disabled KV cache for Phi model on CPU to avoid errors")

    if is_training:
        model.gradient_checkpointing_enable()

        # For CPU-only mode, skip prepare_model_for_kbit_training if not using quantization
        if not (cpu_only and not (load_in_8bit or load_in_4bit)):
            model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)
        if should_print:
            print_trainable_parameters(model)
        if use_accelerate:
            model = accelerator.prepare_model(model)

    return model


def load_pretrained_model(
    ft_model_dir: Path,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    cpu_only: bool = False,
) -> AutoModelForCausalLM:
    """Load the finetuned model saved in the output directory.

    Args:
        ft_model_dir: The path to the finetuned model directory.
        load_in_4bit: Whether to load the model in 4-bit mode.
        load_in_8bit: Whether to load the model in 8-bit mode.
        cpu_only: Whether to force using CPU only and disable quantization.

    Returns:
        The finetuned model.
    """
    import logging

    from transformers import BitsAndBytesConfig

    logger = logging.getLogger(__name__)

    # Explicitly disable MPS when in CPU-only mode
    if cpu_only:
        import os

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        # Force PyTorch to not use MPS
        torch._C._set_mps_enabled(False) if hasattr(
            torch._C, "_set_mps_enabled"
        ) else None
        # Set default device to CPU explicitly
        torch.set_default_device("cpu")
        logger.warning("Disabled MPS device for CPU-only mode.")

    # Set device map based on available hardware and settings
    if cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    # Only use BitsAndBytes config if quantization is requested and we're not in CPU-only mode
    if (
        (load_in_8bit or load_in_4bit)
        and not cpu_only
        and torch.cuda.is_available()
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Use half precision for CPU to reduce memory usage
    torch_dtype = torch.float16 if device_map == "cpu" else None

    # Special config for Phi models on CPU to avoid cache issues
    # Check if it's a Phi model
    is_phi_model = "phi" in str(ft_model_dir).lower()

    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        # Use low_cpu_mem_usage for CPU to minimize memory usage
        "low_cpu_mem_usage": device_map == "cpu",
    }

    # Add special config for Phi models on CPU to avoid cache issues
    if is_phi_model and (cpu_only or device_map == "cpu"):
        logger.warning(
            "Loading Phi model on CPU with special configuration to avoid caching issues"
        )
        model_kwargs["use_flash_attention_2"] = False
        # Set attn_implementation to eager for Phi models on CPU
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(ft_model_dir, **model_kwargs)

    # For Phi models on CPU, disable kv cache feature to avoid errors
    if is_phi_model and (cpu_only or device_map == "cpu"):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
            logger.warning(
                "Disabled KV cache for Phi model on CPU to avoid errors"
            )

    return model
