from pydantic import BaseModel
from typing import Optional, Dict
from zenml.utils.secret_utils import SecretField


class HuggingFaceBaseConfig(BaseModel):
    endpoint_name: Optional[str] = None
    repository: Optional[str] = None
    framework: str
    accelerator: str
    instance_size: str
    instance_type: str
    region: str
    vendor: str
    token: Optional[str] = None
    account_id: Optional[str] = None
    min_replica: Optional[int] = 0
    max_replica: Optional[int] = 1
    revision: Optional[str] = None
    task: Optional[str] = None
    custom_image: Optional[Dict] = None
    namespace: Optional[str] = None
    endpoint_type: str = "public"
