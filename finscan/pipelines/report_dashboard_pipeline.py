from zenml import pipeline
from typing import Dict, Any
from steps.report_dashboard import financial_dashboard

@pipeline
def report_dashboard_pipeline(metrics: Dict[str, Any]):
    """
    ZenML pipeline that orchestrates document loading, processing, and storing.
    """
    financial_dashboard(response=metrics)