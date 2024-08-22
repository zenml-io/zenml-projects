import os

from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_and_push_data_to_s3(bucket_name: str):
    """
    Loads markdown files from the local data/ directory and pushes them to S3.

    Args:
        bucket_name (str): The name of the S3 bucket to push the files to.

    Steps:
        1. Load all markdown files from the data/ directory and its subdirectories.
        2. Connect to S3 and create the specified bucket if it doesn't exist.
        3. Upload all the markdown files to the S3 bucket.

    Returns:
        None
    """
    # Load data (any .md files inside the data/ dir + subdirs)
    data_dir = "data"
    logger.info("Loading markdown files from %s", data_dir)
    md_files = []
    for root, dirs, files in os.walk(data_dir):
        md_files.extend(
            os.path.join(root, file) for file in files if file.endswith(".md")
        )
    logger.info("Found %d markdown files", len(md_files))

    # Create or get an S3 bucket
    logger.info("Connecting to S3")
    zc = Client()
    sc_client = zc.get_service_connector_client(
        name_id_or_prefix="0b04bcae-efc9-4044-a1c2-b86281cb0820",
        resource_type="aws-generic",
    ).connect()
    s3 = sc_client.resource("s3")
    bucket = s3.Bucket(bucket_name)
    if not bucket.creation_date:
        logger.info("Creating S3 bucket %s", bucket_name)
        bucket.create()
    else:
        logger.info("Using existing S3 bucket %s", bucket_name)

    # Push all the data to the S3 bucket
    logger.info("Uploading markdown files to S3 bucket %s", bucket_name)
    for file_path in md_files:
        file_name = os.path.relpath(file_path, ".")
        logger.info("Uploading %s", file_name)
        bucket.upload_file(file_path, file_name)

    logger.info("Finished load_and_push_data_to_s3 step")
