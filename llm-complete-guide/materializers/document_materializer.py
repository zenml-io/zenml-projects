import contextlib
import json
import os
import tempfile
from typing import Any, Dict, Type

import numpy as np
from structures import Document
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


class DocumentMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Document,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Document]) -> Document:
        """Read from artifact store."""
        with fileio.open(os.path.join(self.uri, "page_content.txt"), "r") as f:
            page_content = f.read()

        with fileio.open(os.path.join(self.uri, "metadata.json"), "r") as f:
            metadata = json.load(f)

        embedding = None
        embedding_path = os.path.join(self.uri, "embedding.npy")
        if fileio.exists(embedding_path):
            with contextlib.suppress(IOError, ValueError):
                with fileio.open(embedding_path, "rb") as f:
                    embedding = np.load(f)
        return Document(
            page_content=page_content,
            filename=metadata.get("filename"),
            parent_section=metadata.get("parent_section"),
            url=metadata.get("url"),
            embedding=embedding,
            token_count=metadata.get("token_count"),
        )

    def save(self, document: Document) -> None:
        """Write to artifact store.

        Args:
            document: Document object to save.
        """
        # Create the directory path if it doesn't exist
        fileio.makedirs(self.uri)

        with fileio.open(os.path.join(self.uri, "page_content.txt"), "w") as f:
            f.write(document.page_content)

        metadata = {
            "filename": document.filename,
            "parent_section": document.parent_section,
            "url": document.url,
            "token_count": document.token_count,
        }
        with fileio.open(os.path.join(self.uri, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        if document.embedding is not None:
            with tempfile.NamedTemporaryFile(
                suffix=".npy", delete=False
            ) as temp_file:
                np.save(temp_file.name, document.embedding)
                fileio.copy(
                    temp_file.name, os.path.join(self.uri, "embedding.npy")
                )

    def save_visualizations(
        self, document: Document
    ) -> Dict[str, VisualizationType]:
        """Save visualizations of the document."""
        visualization_uri = os.path.join(self.uri, "visualization.txt")
        with fileio.open(visualization_uri, "w") as f:
            f.write(document.page_content)
        return {visualization_uri: VisualizationType.TEXT}

    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from the document."""
        return {
            "filename": document.filename,
            "parent_section": document.parent_section,
            "url": document.url,
            "token_count": document.token_count,
        }
