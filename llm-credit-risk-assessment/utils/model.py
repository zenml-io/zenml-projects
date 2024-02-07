from typing import Union
from transformers.deepspeed import is_deepspeed_zero3_enabled
from deepspeed.runtime.engine import DeepSpeedEngine
import torch

def get_model_param_count(model: Union[DeepSpeedEngine, torch.nn.Module], trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    if is_deepspeed_zero3_enabled() and isinstance(model, DeepSpeedEngine):
        def numel(p):
            return p.ds_numel

    else:
        def numel(p):
            return p.numel()

    return sum(numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)