import argparse

from pipelines.data_analysis_pipeline import data_analysis_pipeline
from pipelines.training_pipelines import training_pipeline
from steps.data_process import drop_cols, encode_cat_cols
from steps.data_splitter import data_splitter
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.trainer import model_trainer
from steps.visualizer import (
    visualize_statistics,
    visualize_train_test_statistics,
)


def analyze_pipeline():
    """Pipeline for analyzing data."""
    analyze = data_analysis_pipeline(
        ingest_data=ingest_data(),
        data_splitter=data_splitter(),
    )
    analyze.run()
    visualize_statistics()
    visualize_train_test_statistics()


def training_pipeline_run():
    """Pipeline for processing data."""
    train_pipeline = training_pipeline(
        ingest_data=ingest_data(),
        encode_cat_cols=encode_cat_cols(),
        drop_cols=drop_cols(),
        data_splitter=data_splitter(),
        model_trainer=model_trainer(),
        evaluator=evaluation(),
    )
    train_pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=str, choices=["analyze", "train"])
    args = parser.parse_args()
    if args.pipeline == "analyze":
        analyze_pipeline()
    elif args.pipeline == "train":
        training_pipeline_run()
