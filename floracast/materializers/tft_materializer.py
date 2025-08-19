"""
Custom materializer for Darts TFT model objects using io_utils approach.
"""

import os
import tempfile
import pickle
import json
import pandas as pd
import numpy as np
import torch
from typing import Type, Any, Dict
from darts.models import TFTModel
from darts import TimeSeries
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from zenml.metadata.metadata_types import MetadataType
from zenml.io import fileio
from zenml.logger import get_logger

logger = get_logger(__name__)


class TFTModelMaterializer(BaseMaterializer):
    """Materializer for Darts TFT model objects using io_utils pattern."""

    # Import at class level to ensure proper registration
    ASSOCIATED_TYPES = (TFTModel,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[Any]) -> Any:
        """Load a TFT model using enhanced reconstruction strategy."""
        # using top-level TFTModel import

        # Set PyTorch default dtype to float32 for consistent precision
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        try:
            # Check what save strategies were used
            strategy_info = self._load_strategy_info()

            # Try enhanced reconstruction if PyTorch state was saved
            if strategy_info.get("pytorch_model_saved", False):
                try:
                    return self._load_with_pytorch_state()
                except Exception as e:
                    logger.warning(f"Enhanced reconstruction failed: {e}")

            # Fallback to pickle loading
            try:
                return self._load_pickle_format()
            except Exception as e:
                logger.error(f"All loading strategies failed: {e}")
                raise
        finally:
            # Restore original PyTorch dtype
            torch.set_default_dtype(original_dtype)

    def _load_native_format(self) -> Any:
        """Load TFT model using native Darts save format."""
        # using top-level TFTModel import

        # Use temporary directory for native loading
        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_dir = os.path.join(temp_dir, "tft_model")
            os.makedirs(local_model_dir)

            # Download all files from the tft_model directory in artifact store
            tft_dir = os.path.join(self.uri, "tft_model")

            # List all files in the tft_model directory and download them
            try:
                # Try to list files in the tft_model directory
                files_to_copy = []
                try:
                    # Try to use fileio to list files if supported
                    for filename in [
                        "model.pth.tar",
                        "model_params.pkl",
                        "model.pkl",
                        "checkpoint.pth",
                    ]:
                        src_path = os.path.join(tft_dir, filename)
                        if fileio.exists(src_path):
                            files_to_copy.append(filename)
                except Exception:
                    # Fallback to known essential files
                    files_to_copy = ["model.pth.tar", "model_params.pkl"]

                # Copy each file
                for filename in files_to_copy:
                    src_path = os.path.join(tft_dir, filename)
                    dst_path = os.path.join(local_model_dir, filename)

                    try:
                        with fileio.open(src_path, "rb") as src_f:
                            with open(dst_path, "wb") as dst_f:
                                dst_f.write(src_f.read())
                        logger.info(f"Downloaded {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to download {filename}: {e}")
                        continue

                # Load using TFT's native method
                logger.info("Loading TFT model from native format")
                model = TFTModel.load(local_model_dir)
                logger.info("Successfully loaded TFT model from native format")
                return model

            except Exception as e:
                logger.warning(f"Native format loading failed: {e}")
                raise

    def _load_strategy_info(self) -> dict:
        """Load information about what save strategies were used."""
        try:
            # using top-level json import
            info_path = os.path.join(self.uri, "save_info.json")
            with fileio.open(info_path, "r") as f:
                return json.load(f)
        except Exception:
            # Default to old behavior if no strategy info found
            return {"pytorch_model_saved": False}

    def _load_with_pytorch_state(self) -> Any:
        """Load TFT model with PyTorch state reconstruction."""
        # using top-level TFTModel, torch, json imports

        logger.info("Loading TFT model with PyTorch state reconstruction")

        # Load the pickle model first to get the structure
        pickle_path = os.path.join(self.uri, "model.pkl")
        with fileio.open(pickle_path, "rb") as f:
            model = pickle.load(f)

        # Load the saved PyTorch state
        state_dict_path = os.path.join(self.uri, "pytorch_state.pth")
        with tempfile.NamedTemporaryFile() as temp_file:
            with fileio.open(state_dict_path, "rb") as src_f:
                with open(temp_file.name, "wb") as dst_f:
                    dst_f.write(src_f.read())

            # Load the state dict
            try:
                state_dict = torch.load(temp_file.name, map_location="cpu")
            except Exception as e:
                logger.warning(f"Failed to load PyTorch state dict: {e}")
                # Try with different loading strategy
                state_dict = torch.load(
                    temp_file.name, map_location="cpu", weights_only=False
                )

        # Load TFT metadata to reconstruct the model properly
        metadata_path = os.path.join(self.uri, "tft_metadata.json")
        with fileio.open(metadata_path, "r") as f:
            metadata = json.load(f)

        # If the internal model is None, we need to create it
        if getattr(model, "model", None) is None:
            logger.info("Reconstructing internal PyTorch model")

            # Create a new model with same parameters to get the architecture
            temp_model = TFTModel(**metadata.get("model_params", {}))

            # Create minimal training data to initialize the model architecture
            # using top-level pandas, numpy, TimeSeries imports

            # Create dummy training data based on saved metadata
            metadata.get("training_series_info", {})
            dummy_length = max(
                temp_model.input_chunk_length
                + temp_model.output_chunk_length
                + 5,
                50,
            )

            dates = pd.date_range("2020-01-01", periods=dummy_length, freq="D")
            values = np.random.randn(dummy_length).astype(np.float32)
            dummy_series = TimeSeries.from_dataframe(
                pd.DataFrame({"ds": dates, "y": values}),
                time_col="ds",
                value_cols="y",
            ).astype(np.float32)

            # Partially fit to create the internal model structure
            temp_model.fit(dummy_series, epochs=1, verbose=False)

            # Now load the saved state into the reconstructed model
            if hasattr(temp_model, "model") and temp_model.model is not None:
                temp_model.model.load_state_dict(state_dict)

                # Replace the internal model in our loaded model
                model.model = temp_model.model
                model._fit_called = True

                logger.debug(
                    "Successfully reconstructed TFT model with saved PyTorch state"
                )
            else:
                logger.warning("Failed to create internal model structure")

        else:
            # Model structure exists, just load the state
            model.model.load_state_dict(state_dict)

        return model

    def _load_pickle_format(self) -> Any:
        """Fallback to pickle loading."""
        pickle_path = os.path.join(self.uri, "model.pkl")

        logger.info("Loading TFT model from pickle format")
        with fileio.open(pickle_path, "rb") as f:
            model = pickle.load(f)

            logger.warning(
                "Loaded from pickle - internal PyTorch model may be None"
            )
            return model

    def save(self, data: Any) -> None:
        """Save TFT model using enhanced strategy that preserves internal PyTorch model."""
        logger.info("Saving TFT model using enhanced strategy")

        # Strategy 1: Save PyTorch model state separately if model is fitted and has internal model
        pytorch_model_saved = False
        if (
            getattr(data, "_fit_called", False)
            and getattr(data, "model", None) is not None
        ):
            try:
                self._save_pytorch_state(data)
                pytorch_model_saved = True
                logger.info("Successfully saved PyTorch model state")
            except Exception as e:
                logger.warning(f"PyTorch state save failed: {e}")

        # Strategy 2: Save metadata and TFT attributes
        try:
            self._save_tft_metadata(data)
            logger.info("Successfully saved TFT metadata")
        except Exception as e:
            logger.warning(f"TFT metadata save failed: {e}")

        # Strategy 3: Always save pickle as backup (but will need reconstruction on load)
        try:
            self._save_pickle_format(data)
            logger.info("Successfully saved TFT model as pickle backup")
        except Exception as e:
            logger.warning(f"Pickle backup save failed: {e}")

        # Create a flag file to indicate which strategies succeeded
        strategy_info = {
            "pytorch_model_saved": pytorch_model_saved,
            "fit_called": getattr(data, "_fit_called", False),
            "has_internal_model": getattr(data, "model", None) is not None,
        }
        try:
            info_path = os.path.join(self.uri, "save_info.json")
            with fileio.open(info_path, "w") as f:
                json.dump(strategy_info, f)
        except Exception as e:
            logger.warning(f"Failed to save strategy info: {e}")

    def _save_native_format(self, data: Any) -> None:
        """Save TFT model using native Darts format."""
        # Check if the model is fitted (has trained internal model)
        if (
            not getattr(data, "_fit_called", False)
            or getattr(data, "model", None) is None
        ):
            logger.warning(
                "TFT model not fitted or has no internal model - skipping native save"
            )
            raise ValueError(
                "TFT model must be fitted before saving in native format"
            )

        # Use temporary directory for native saving
        with tempfile.TemporaryDirectory() as temp_dir:
            local_model_dir = os.path.join(temp_dir, "tft_model")

            # Save using TFT's native method
            data.save(local_model_dir)

            # Check if any files were actually created
            files_created = []
            if os.path.exists(local_model_dir):
                for root, dirs, files in os.walk(local_model_dir):
                    for filename in files:
                        files_created.append(filename)

            if not files_created:
                logger.warning(
                    "No files created by TFT native save - model may not be fitted"
                )
                raise ValueError("TFT native save created no files")

            logger.info(f"TFT native save created files: {files_created}")

            # Upload all files to artifact store
            tft_dir = os.path.join(self.uri, "tft_model")

            for root, dirs, files in os.walk(local_model_dir):
                for filename in files:
                    src_path = os.path.join(root, filename)
                    # Calculate relative path from local_model_dir
                    rel_path = os.path.relpath(src_path, local_model_dir)
                    dst_path = os.path.join(tft_dir, rel_path)

                    # Create directory structure if needed
                    dst_dir = os.path.dirname(dst_path)
                    if dst_dir and dst_dir != tft_dir:
                        # Create intermediate directories
                        parts = os.path.relpath(dst_dir, self.uri).split(
                            os.sep
                        )
                        current_path = self.uri
                        for part in parts:
                            current_path = os.path.join(current_path, part)
                            # fileio doesn't have makedirs, but we can try to write to nested paths
                            pass

                    # Copy file to artifact store using fileio
                    with open(src_path, "rb") as src_f:
                        with fileio.open(dst_path, "wb") as dst_f:
                            dst_f.write(src_f.read())

                    logger.info(f"Uploaded {rel_path} to artifact store")

    def _save_pytorch_state(self, data: Any) -> None:
        """Save the internal PyTorch model state dict separately."""
        # using top-level torch import

        if hasattr(data, "model") and data.model is not None:
            # Save the PyTorch model state dict
            state_dict_path = os.path.join(self.uri, "pytorch_state.pth")
            with tempfile.NamedTemporaryFile() as temp_file:
                torch.save(data.model.state_dict(), temp_file.name)
                with open(temp_file.name, "rb") as src_f:
                    with fileio.open(state_dict_path, "wb") as dst_f:
                        dst_f.write(src_f.read())

            # Also save the model architecture if available
            if hasattr(data, "model_params"):
                arch_path = os.path.join(self.uri, "model_arch.pkl")
                with tempfile.NamedTemporaryFile() as temp_file:
                    # using top-level pickle import
                    with open(temp_file.name, "wb") as f:
                        pickle.dump(data.model_params, f)
                    with open(temp_file.name, "rb") as src_f:
                        with fileio.open(arch_path, "wb") as dst_f:
                            dst_f.write(src_f.read())

    def _save_tft_metadata(self, data: Any) -> None:
        """Save TFT model metadata and configuration."""
        # using top-level json import

        # Extract key TFT attributes that we need to reconstruct
        metadata = {
            "input_chunk_length": getattr(data, "input_chunk_length", None),
            "output_chunk_length": getattr(data, "output_chunk_length", None),
            "fit_called": getattr(data, "_fit_called", False),
            "model_params": getattr(data, "_model_params", {}),
            "training_series_info": None,
        }

        # Save training series information if available
        if (
            hasattr(data, "training_series")
            and data.training_series is not None
        ):
            ts = data.training_series
            metadata["training_series_info"] = {
                "length": len(ts),
                "width": ts.width,
                "start_time": str(ts.start_time()),
                "end_time": str(ts.end_time()),
                "freq": str(ts.freq) if ts.freq else None,
                "columns": list(ts.columns) if hasattr(ts, "columns") else [],
            }

        # Convert any non-serializable values
        def make_serializable(obj):
            if hasattr(obj, "__dict__"):
                return str(obj)
            return obj

        # Clean metadata for JSON serialization
        clean_metadata = {}
        for key, value in metadata.items():
            try:
                json.dumps(value)  # Test if serializable
                clean_metadata[key] = value
            except (TypeError, ValueError):
                clean_metadata[key] = str(value)

        metadata_path = os.path.join(self.uri, "tft_metadata.json")
        with fileio.open(metadata_path, "w") as f:
            json.dump(clean_metadata, f, indent=2)

    def _save_pickle_format(self, data: Any) -> None:
        """Save TFT model using pickle format as backup."""
        pickle_path = os.path.join(self.uri, "model.pkl")

        with fileio.open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    def extract_metadata(self, data: Any) -> Dict[str, MetadataType]:
        """Extract metadata from TFT model."""
        metadata = {
            "input_chunk_length": data.input_chunk_length,
            "output_chunk_length": data.output_chunk_length,
            "model_name": getattr(data, "model_name", "TFTModel"),
            "n_epochs": getattr(data, "n_epochs", "unknown"),
            "fit_called": getattr(data, "_fit_called", False),
            "has_internal_model": getattr(data, "model", None) is not None,
        }

        if hasattr(data, "_model_params"):
            metadata.update(
                {
                    "hidden_size": data._model_params.get(
                        "hidden_size", "unknown"
                    ),
                    "lstm_layers": data._model_params.get(
                        "lstm_layers", "unknown"
                    ),
                    "dropout": data._model_params.get("dropout", "unknown"),
                }
            )

        return metadata
