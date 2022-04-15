from pipelines.data_analysis_pipeline import data_analysis_pipeline
from steps.data_splitter import data_splitter
from steps.ingest_data import ingest_data
from steps.visualizer import (
    visualize_statistics,
    visualize_train_test_statistics,
)


def main():
    analyze = data_analysis_pipeline(
        ingest_data(),
        visualize_statistics(),
        data_splitter(),
        visualize_train_test_statistics(),
    )
    analyze.run()


if __name__ == "__main__":
    main()
