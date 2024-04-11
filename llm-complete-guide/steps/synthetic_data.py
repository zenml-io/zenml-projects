from typing import List

from structures import Document
from zenml import step
from zenml.client import Client


@step
def generate_questions_from_chunks() -> List[Document]:
    """Generate questions from chunks."""
    client = Client()
    docs_with_embeddings = client.get_artifact_version(
        name_id_or_prefix="documents_with_embeddings"
    )
    breakpoint()
