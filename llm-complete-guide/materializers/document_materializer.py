import json
import os
from typing import Any, Dict, List, Type

import numpy as np
from structures import Document
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


class DocumentMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (List[Document],)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[List[Document]]) -> List[Document]:
        """Read from artifact store."""
        documents = []
        for file_name in fileio.listdir(self.uri):
            if file_name.startswith("document_") and file_name.endswith(
                ".json"
            ):
                with fileio.open(os.path.join(self.uri, file_name), "r") as f:
                    document_data = json.load(f)
                    document = Document(
                        page_content=document_data["page_content"],
                        filename=document_data["filename"],
                        parent_section=document_data["parent_section"],
                        url=document_data["url"],
                        embedding=(
                            np.array(document_data["embedding"])
                            if document_data["embedding"]
                            else None
                        ),
                        token_count=document_data["token_count"],
                    )
                    documents.append(document)
        return documents

    def save(self, documents: List[Document]) -> None:
        """Write to artifact store."""
        for i, document in enumerate(documents):
            document_data = {
                "page_content": document.page_content,
                "filename": document.filename,
                "parent_section": document.parent_section,
                "url": document.url,
                "embedding": (
                    document.embedding.tolist()
                    if document.embedding is not None
                    else None
                ),
                "token_count": document.token_count,
            }
            with fileio.open(
                os.path.join(self.uri, f"document_{i}.json"), "w"
            ) as f:
                json.dump(document_data, f)

    def save_visualizations(
        self, documents: List[Document]
    ) -> Dict[str, VisualizationType]:
        """Save visualizations of the documents."""
        visualization_uris = {}
        for i, document in enumerate(documents):
            visualization_uri = os.path.join(
                self.uri, f"visualization_{i}.txt"
            )
            with fileio.open(visualization_uri, "w") as f:
                f.write(document.page_content)
            visualization_uris[visualization_uri] = VisualizationType.TEXT
        return visualization_uris

    def extract_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract metadata from the documents."""
        metadata = {
            "num_documents": len(documents),
            "total_tokens": sum(
                doc.token_count
                for doc in documents
                if doc.token_count is not None
            ),
        }
        return metadata
