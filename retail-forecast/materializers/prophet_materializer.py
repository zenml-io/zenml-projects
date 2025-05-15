import json
import os
from typing import Any, Dict, Type

from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer


class ProphetMaterializer(BaseMaterializer):
    """Materializer for Prophet models."""

    ASSOCIATED_TYPES = (Prophet, dict)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[Any]) -> Any:
        """Load a Prophet model or dictionary of Prophet models from storage.

        Args:
            data_type: The data type to load.

        Returns:
            A Prophet model or dictionary of Prophet models.
        """
        # Check if we're loading a dictionary
        if data_type == dict:
            # Path to the keys file
            keys_path = os.path.join(self.uri, "keys.json")

            # Load the keys
            with self.artifact_store.open(keys_path, "r") as f:
                keys = json.load(f)

            # Load each model
            result = {}
            for key in keys:
                model_dir = os.path.join(self.uri, "models", key)
                model_path = os.path.join(model_dir, "model.json")

                # Load the model JSON
                with self.artifact_store.open(model_path, "r") as f:
                    model_json = f.read()

                # Create the Prophet model
                result[key] = model_from_json(model_json)

            return result
        else:
            # Path to the serialized model
            model_path = os.path.join(self.uri, "model.json")

            # Load the serialized model
            with self.artifact_store.open(model_path, "r") as f:
                model_json = f.read()

            # Create a new Prophet model from the JSON
            model = model_from_json(model_json)

            return model

    def save(self, data: Any) -> None:
        """Save a Prophet model or dictionary of Prophet models to storage.

        Args:
            data: The Prophet model or dictionary of Prophet models to save.
        """
        # Check if we're saving a dictionary
        if isinstance(data, dict):
            # First check if the dictionary contains Prophet models
            if not all(isinstance(model, Prophet) for model in data.values()):
                # If this is just a regular dictionary, use the default dictionary materializer
                # by raising a ValueError
                raise ValueError(
                    "This materializer only supports dictionaries of Prophet models."
                )

            # Save the keys
            keys_path = os.path.join(self.uri, "keys.json")
            with self.artifact_store.open(keys_path, "w") as f:
                json.dump(list(data.keys()), f)

            # Save each model
            for key, model in data.items():
                # Create a directory for this model
                model_dir = os.path.join(self.uri, "models", key)
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)

                # Serialize the model to JSON
                model_json = model_to_json(model)

                # Save the serialized model
                model_path = os.path.join(model_dir, "model.json")
                with self.artifact_store.open(model_path, "w") as f:
                    f.write(model_json)
        else:
            # Path to save the serialized model
            model_path = os.path.join(self.uri, "model.json")

            # Serialize the model to JSON
            model_json = model_to_json(data)

            # Save the serialized model
            with self.artifact_store.open(model_path, "w") as f:
                f.write(model_json)

    def extract_metadata(self, data: Any) -> Dict[str, Any]:
        """Extract metadata from the Prophet model or dictionary of Prophet models.

        Args:
            data: The Prophet model or dictionary of Prophet models to extract metadata from.

        Returns:
            A dictionary of metadata.
        """
        # Check if we're extracting metadata from a dictionary
        if isinstance(data, dict):
            # Extract metadata for each model
            models_metadata = {}
            for key, model in data.items():
                models_metadata[key] = self._extract_model_metadata(model)

            metadata = {
                "model_type": "prophet_dictionary",
                "num_models": len(data),
                "models": models_metadata,
            }
            return metadata
        else:
            # Extract metadata for a single model
            return self._extract_model_metadata(data)

    def _extract_model_metadata(self, model: Prophet) -> Dict[str, Any]:
        """Extract metadata from a single Prophet model.

        Args:
            model: The Prophet model to extract metadata from.

        Returns:
            A dictionary of metadata.
        """
        metadata = {
            "model_type": "prophet",
            "seasonality_mode": model.seasonality_mode,
            "growth": model.growth,
        }

        # Add information about seasonalities if available
        if hasattr(model, "seasonalities") and model.seasonalities:
            metadata["seasonalities"] = list(model.seasonalities.keys())

        # Add information about regressors if available
        if hasattr(model, "extra_regressors") and model.extra_regressors:
            metadata["regressors"] = len(model.extra_regressors)

        return metadata
