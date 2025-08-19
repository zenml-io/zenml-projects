"""
Custom materializer for Darts TimeSeries objects.
"""

import os
from typing import Type, Any, Dict

from darts import TimeSeries
from zenml.enums import ArtifactType, VisualizationType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.metadata.metadata_types import MetadataType
from zenml.io import fileio


class DartsTimeSeriesMaterializer(BaseMaterializer):
    """Materializer for Darts TimeSeries objects."""
    
    ASSOCIATED_TYPES = (TimeSeries,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA
    
    def load(self, data_type: Type[Any]) -> TimeSeries:
        """Load a Darts TimeSeries from CSV format."""
        filepath = os.path.join(self.uri, "timeseries.csv")
        
        with fileio.open(filepath, "r") as f:
            import pandas as pd
            df = pd.read_csv(f)
            
            # Assume first column is datetime index and rest are values
            time_col = df.columns[0]
            value_cols = df.columns[1:].tolist()
            
            df[time_col] = pd.to_datetime(df[time_col])
            
            series = TimeSeries.from_dataframe(
                df,
                time_col=time_col,
                value_cols=value_cols
            )
        
        return series
    
    def save(self, data: TimeSeries) -> None:
        """Save a Darts TimeSeries to CSV format."""
        filepath = os.path.join(self.uri, "timeseries.csv")
        
        # Convert to DataFrame first to control CSV formatting
        df = data.pd_dataframe()
        df.reset_index(inplace=True)
        
        with fileio.open(filepath, "w") as f:
            df.to_csv(f, index=False)
    
    def save_visualizations(self, data: TimeSeries) -> Dict[str, VisualizationType]:
        """Generate HTML visualization for TimeSeries."""
        import matplotlib.pyplot as plt
        import io
        import base64
        
        try:
            # Create a simple line plot
            fig, ax = plt.subplots(figsize=(12, 6))
            data.plot(ax=ax)
            ax.set_title(f'Time Series ({len(data)} points)')
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Create HTML with embedded image
            html_content = f"""
            <div style="text-align: center; padding: 20px;">
                <h3>Time Series Visualization</h3>
                <p>Series length: {len(data)} points</p>
                <p>Time range: {data.start_time()} to {data.end_time()}</p>
                <img src="data:image/png;base64,{image_base64}" 
                     style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px;">
            </div>
            """
            
            # Save HTML visualization
            viz_path = os.path.join(self.uri, "visualization.html")
            with fileio.open(viz_path, "w") as f:
                f.write(html_content)
            
            return {viz_path: VisualizationType.HTML}
            
        except Exception as e:
            # Fallback to no visualization if plotting fails
            return {}
    
    def extract_metadata(self, data: TimeSeries) -> Dict[str, MetadataType]:
        """Extract metadata about the TimeSeries."""
        metadata = {
            "length": len(data),
            "start_time": str(data.start_time()),
            "end_time": str(data.end_time()),
            "frequency": str(data.freq) if data.freq else "unknown",
            "columns": list(data.columns) if data.width > 1 else [data.columns[0]],
            "n_components": data.width,
        }
        
        # Add basic statistics if numeric
        try:
            values = data.values()
            metadata.update({
                "min_value": float(values.min()),
                "max_value": float(values.max()),
                "mean_value": float(values.mean()),
                "std_value": float(values.std()),
            })
        except Exception:
            # Skip statistics if not numeric
            pass
        
        return metadata