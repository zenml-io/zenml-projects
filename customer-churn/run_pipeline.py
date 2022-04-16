from materializer.customer_materializer import cs_materializer
from pipelines.data_analysis_pipeline import data_analysis_pipeline
from steps.data_splitter import data_splitter
from steps.ingest_data import ingest_data
from steps.visualizer import (
    visualize_statistics,
    visualize_train_test_statistics,
)


def analyze_pipeline():
    """Pipeline for analyzing data."""
    analyze = data_analysis_pipeline(
        ingest_data(),
        data_splitter(),
    )
    analyze.run()
    visualize_statistics()
    visualize_train_test_statistics()


if __name__ == "__main__":
    analyze_pipeline()
