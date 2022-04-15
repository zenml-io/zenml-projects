from pipelines.data_analysis_pipeline import data_analysis_pipeline
from steps.ingest_data import data_splitter, ingest_data
from steps.visualizer import visualize_statistics, visualize_whylabs_statistics


def main():
    analyze = data_analysis_pipeline(
        ingest_data(),
        data_splitter(),
        visualize_statistics(),
        visualize_whylabs_statistics(),
    )
    analyze.run()


if __name__ == "__main__":
    main()
