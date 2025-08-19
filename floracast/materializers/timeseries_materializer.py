"""
Custom materializer for Darts TimeSeries objects.

This materializer saves a `darts.TimeSeries` as:
- series.csv: the time-indexed values
- metadata.json: minimal reconstruction metadata (freq, columns, etc.)
- static_covariates.csv (optional): static covariates if present
"""

import os
import json
import tempfile
from typing import Any, Dict, Type

import pandas as pd
import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from darts import TimeSeries
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from zenml.metadata.metadata_types import MetadataType
from zenml.io import fileio
from zenml.logger import get_logger


logger = get_logger(__name__)


class DartsTimeSeriesMaterializer(BaseMaterializer):
    """Materializer for Darts TimeSeries objects."""

    ASSOCIATED_TYPES = (TimeSeries,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Any]) -> Any:
        """Load a Darts TimeSeries from CSV + metadata."""
        metadata_path = os.path.join(self.uri, "metadata.json")
        series_path = os.path.join(self.uri, "series.csv")
        static_covariates_path = os.path.join(
            self.uri, "static_covariates.csv"
        )

        # Read metadata
        metadata = {}
        if fileio.exists(metadata_path):
            with fileio.open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            logger.warning(
                "metadata.json not found. Proceeding with best-effort defaults."
            )

        # Read series
        if not fileio.exists(series_path):
            raise FileNotFoundError(
                "series.csv not found for TimeSeries artifact"
            )

        with fileio.open(series_path, "r") as f:
            df = pd.read_csv(f)

        time_col = metadata.get("time_col", "time")
        value_cols = metadata.get("value_cols")
        freq = metadata.get("freq")
        time_index_type = metadata.get("time_index_type")
        time_tz = metadata.get("time_tz")
        dtypes_map = metadata.get("dtypes") or {}

        if value_cols is None:
            # Default to all non-time columns as values
            value_cols = [c for c in df.columns if c != time_col]

        # Ensure proper dtype restoration
        for col, dtype_str in dtypes_map.items():
            if col in df.columns and col != time_col:
                try:
                    df[col] = df[col].astype(dtype_str)
                except Exception:
                    pass

        # Ensure datetime type for the time column with timezone if present
        if time_col in df.columns:
            try:
                df[time_col] = pd.to_datetime(df[time_col], utc=False)
                if time_tz:
                    try:
                        # If parsed as timezone-aware, convert; otherwise localize
                        if df[time_col].dt.tz is not None:
                            df[time_col] = df[time_col].dt.tz_convert(time_tz)
                        else:
                            df[time_col] = df[time_col].dt.tz_localize(time_tz)
                    except Exception:
                        pass
            except Exception:
                pass

        ts = TimeSeries.from_dataframe(
            df, time_col=time_col, value_cols=value_cols, freq=freq
        )

        # Restore static covariates if present
        if fileio.exists(static_covariates_path):
            with fileio.open(static_covariates_path, "r") as f:
                sc_df = pd.read_csv(f)
            try:
                ts = ts.with_static_covariates(sc_df)
            except Exception as e:
                logger.warning(f"Failed to attach static covariates: {e}")

        return ts

    def save(self, data: Any) -> None:
        """Save a Darts TimeSeries to CSV + metadata.

        We avoid pickling for portability and artifact store compatibility.
        """
        if not isinstance(data, TimeSeries):
            raise TypeError(
                "DartsTimeSeriesMaterializer can only handle darts.TimeSeries instances"
            )

        # Extract dataframe with time index
        df = data.pd_dataframe()
        time_index = df.index
        df_reset = df.reset_index()
        time_col_name = df_reset.columns[0]
        value_cols = list(df_reset.columns[1:])

        # Save series.csv using a temp file to support remote artifact stores
        series_path = os.path.join(self.uri, "series.csv")
        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
            # Write datetime in a stable ISO format including timezone when present
            date_format = "%Y-%m-%dT%H:%M:%S%z"
            try:
                df_reset.to_csv(tmp.name, index=False, date_format=date_format)
            except TypeError:
                # Older pandas may not support date_format for to_csv; fallback
                df_reset.to_csv(tmp.name, index=False)
            with open(tmp.name, "rb") as src_f:
                with fileio.open(series_path, "wb") as dst_f:
                    dst_f.write(src_f.read())

        # Save static covariates if present
        try:
            static_covariates = data.static_covariates
        except Exception:
            static_covariates = None

        if static_covariates is not None:
            sc_path = os.path.join(self.uri, "static_covariates.csv")
            with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
                # Ensure DataFrame
                sc_df = (
                    static_covariates
                    if isinstance(static_covariates, pd.DataFrame)
                    else pd.DataFrame(static_covariates)
                )
                sc_df.to_csv(tmp.name, index=False)
                with open(tmp.name, "rb") as src_f:
                    with fileio.open(sc_path, "wb") as dst_f:
                        dst_f.write(src_f.read())

        # Determine a stable pandas frequency alias (e.g., "D")
        freq_alias = None
        try:
            freq_alias = getattr(time_index, "freqstr", None)
            if freq_alias is None:
                freq_alias = pd.infer_freq(time_index)
        except Exception:
            freq_alias = None

        # Gather meta about time index
        try:
            if hasattr(time_index, "tz") and time_index.tz is not None:
                time_tz = str(time_index.tz)
            else:
                time_tz = None
        except Exception:
            time_tz = None

        if str(type(time_index)).endswith("DatetimeIndex'>"):
            time_index_type = "datetime"
        elif str(type(time_index)).endswith("PeriodIndex'>"):
            time_index_type = "period"
        elif str(type(time_index)).endswith("Int64Index'>"):
            time_index_type = "int"
        else:
            time_index_type = "other"

        # Capture dtypes for value columns
        dtypes_map = {col: str(df_reset[col].dtype) for col in value_cols}

        # Save minimal metadata
        metadata = {
            "time_col": time_col_name,
            "value_cols": value_cols,
            "freq": freq_alias,
            "time_index_type": time_index_type,
            "time_tz": time_tz,
            "dtypes": dtypes_map,
            "length": len(data),
            "is_univariate": data.width == 1,
        }

        metadata_path = os.path.join(self.uri, "metadata.json")
        with fileio.open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create a lightweight preview plot and visualizations bundle
        try:
            # Save a standalone preview image at the artifact root
            preview_path = os.path.join(self.uri, "preview.png")
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                fig, ax = plt.subplots(figsize=(8, 3))
                x_vals = df_reset[time_col_name]
                for col in value_cols:
                    ax.plot(x_vals, df_reset[col], label=col, linewidth=1)
                ax.set_title("TimeSeries Preview")
                ax.set_xlabel(str(time_col_name))
                ax.set_ylabel("value")
                if len(value_cols) > 1:
                    ax.legend(loc="upper right", fontsize=8)
                fig.tight_layout()
                fig.savefig(tmp.name, dpi=120)
                plt.close(fig)
                with open(tmp.name, "rb") as src_f:
                    with fileio.open(preview_path, "wb") as dst_f:
                        dst_f.write(src_f.read())
            logger.info(f"Saved TimeSeries preview to {preview_path}")

            # Also store under a standard 'visualizations' folder with an index.html
            viz_dir = os.path.join(self.uri, "visualizations")
            viz_img_path = os.path.join(viz_dir, "preview.png")
            viz_index_path = os.path.join(viz_dir, "index.html")

            # Write the image
            with fileio.open(preview_path, "rb") as src_f:
                with fileio.open(viz_img_path, "wb") as dst_f:
                    dst_f.write(src_f.read())

            # Write a simple HTML wrapper
            html = (
                "<html><head><meta charset='utf-8'><title>TimeSeries Preview"
                '</title></head><body style="margin:0;padding:0;">'
                "<img src='preview.png' style='max-width:100%;height:auto;'/></body></html>"
            )
            with fileio.open(viz_index_path, "w") as f:
                f.write(html)
        except Exception as e:
            # Best-effort; do not fail the materialization for plotting issues
            logger.warning(f"Failed to create TimeSeries preview plot: {e}")

        # Attempt to let the base implementation pick up visualizations (best-effort)
        try:
            self.save_visualizations(data)  # type: ignore[attr-defined]
        except Exception:
            pass

    def save_visualizations(self, data: Any) -> None:
        """Save standard visualizations that ZenML UIs can render.

        Creates a 'visualizations' directory with a preview image and index.html.
        """
        try:
            df = data.pd_dataframe()
            df_reset = df.reset_index()
            time_col_name = df_reset.columns[0]
            value_cols = list(df_reset.columns[1:])

            viz_dir = os.path.join(self.uri, "visualizations")
            viz_img_path = os.path.join(viz_dir, "preview.png")
            viz_index_path = os.path.join(viz_dir, "index.html")

            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                fig, ax = plt.subplots(figsize=(8, 3))
                x_vals = df_reset[time_col_name]
                for col in value_cols:
                    ax.plot(x_vals, df_reset[col], label=col, linewidth=1)
                ax.set_title("TimeSeries Preview")
                ax.set_xlabel(str(time_col_name))
                ax.set_ylabel("value")
                if len(value_cols) > 1:
                    ax.legend(loc="upper right", fontsize=8)
                fig.tight_layout()
                fig.savefig(tmp.name, dpi=120)
                plt.close(fig)
                with open(tmp.name, "rb") as src_f:
                    with fileio.open(viz_img_path, "wb") as dst_f:
                        dst_f.write(src_f.read())

            html = (
                "<html><head><meta charset='utf-8'><title>TimeSeries Preview"
                '</title></head><body style="margin:0;padding:0;">'
                "<img src='preview.png' style='max-width:100%;height:auto;'/></body></html>"
            )
            with fileio.open(viz_index_path, "w") as f:
                f.write(html)
        except Exception as e:
            logger.warning(f"Failed to write visualizations bundle: {e}")

    def extract_metadata(self, data: Any) -> Dict[str, MetadataType]:
        """Extract lightweight metadata from a TimeSeries."""
        return {
            "length": len(data),
            "width": data.width,
            "start_time": str(data.start_time()),
            "end_time": str(data.end_time()),
            "freq": str(data.freq)
            if getattr(data, "freq", None) is not None
            else "unknown",
            "is_univariate": data.width == 1,
        }
