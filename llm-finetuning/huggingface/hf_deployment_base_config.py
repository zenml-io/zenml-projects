from pydantic import BaseModel
from typing import Optional, Dict
from zenml.utils.secret_utils import SecretField


class HuggingFaceBaseConfig(BaseModel):
    """Huggingface Inference Endpoint configuration."""

    endpoint_name: Optional[str] = ""
    repository: Optional[str] = None
    framework: Optional[str] = None
    accelerator: Optional[str] = None
    instance_size: Optional[str] = None
    instance_type: Optional[str] = None
    region: Optional[str] = None
    vendor: Optional[str] = None
    token: Optional[str] = None
    account_id: Optional[str] = None
    min_replica: Optional[int] = 0
    max_replica: Optional[int] = 1
    revision: Optional[str] = None
    task: Optional[str] = None
    custom_image: Optional[Dict] = None
    namespace: Optional[str] = None
    endpoint_type: str = "public"
