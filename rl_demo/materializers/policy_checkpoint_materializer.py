"""Materializer for PyTorch policy checkpoints (model_state_dict + config)."""

import io
import tempfile
from pathlib import Path

import torch

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


class PolicyCheckpointMaterializer(BaseMaterializer):
    """Materializer for PyTorch policy checkpoints (model_state_dict + config)."""

    ASSOCIATED_TYPES = (dict,)

    def load(self, data_type: type) -> dict:
        """Load checkpoint from artifact store."""
        import os

        path = os.path.join(self.uri, "checkpoint.pt")
        with fileio.open(path, "rb") as f:
            buffer = io.BytesIO(f.read())
        return torch.load(buffer, weights_only=False)

    def save(self, data: dict) -> None:
        """Save checkpoint to artifact store."""
        import os

        fileio.makedirs(self.uri)
        path = os.path.join(self.uri, "checkpoint.pt")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            torch.save(data, tmp.name)
        try:
            fileio.copy(tmp.name, path, overwrite=True)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
