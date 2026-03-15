"""Register custom materializers for the code evaluation pipeline."""

from zenml.materializers.materializer_registry import materializer_registry

from materializers.html_report_materializer import HTMLReportMaterializer
from utils.scoring import CodeEvalReport


def register_materializers() -> None:
    """Register CodeEvalReport materializer with ZenML."""
    materializer_registry.register_and_overwrite_type(
        key=CodeEvalReport, type_=HTMLReportMaterializer
    )
