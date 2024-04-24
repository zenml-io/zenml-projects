import os
from typing import Any, Tuple

import boto3
import sagemaker


def get_aws_config() -> Tuple[Any, sagemaker.Session, str]:
    REGION_NAME = os.getenv("AWS_REGION", "us-east-1")
    ROLE_NAME = os.getenv("AWS_ROLE_NAME", "hamza_connector")
    os.environ["AWS_DEFAULT_REGION"] = REGION_NAME

    auth_arguments = {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", None),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", None),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN", None),
        "region_name": REGION_NAME,
    }
    iam = boto3.client("iam", **auth_arguments)
    role = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    session = sagemaker.Session(boto3.Session(**auth_arguments))

    return role, session, REGION_NAME
