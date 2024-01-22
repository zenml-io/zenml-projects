from pydantic import BaseModel
from typing import Optional, Dict


class HuggingFaceBaseConfig(BaseModel):
    endpoint_name: str
    repository: str
    framework: str
    accelerator: str
    instance_size: str
    instance_type: str
    region: str
    vendor: str
    token: str
    account_id: Optional[str] = None
    min_replica: Optional[int] = 0
    max_replica: Optional[int] = 1
    revision: Optional[str] = None
    task: Optional[str] = None
    custom_image: Optional[Dict] = None
    namespace: Optional[str] = None
    endpoint_type: str = "public"
