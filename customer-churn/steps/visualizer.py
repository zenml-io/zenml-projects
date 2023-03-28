import logging

from zenml.integrations.facets.visualizers.facet_statistics_visualizer import (
    FacetStatisticsVisualizer,
)
from zenml.post_execution import get_pipeline


def visualize_statistics():
    """
    This function will get the last pipeline run, get the last step of the pipeline run, and then
    visualize the statistics of the last step.
    """
    try:
        pipe = get_pipeline("data_analysis_pipeline")
        ingest_data = pipe.runs[0].get_step(step="ingest_data")
        FacetStatisticsVisualizer().visualize(ingest_data)
    except Exception as e:
        logging.error(e)


def visualize_train_test_statistics():
    """
    It visualizes the statistics of the train and test datasets.
    """
    try:
        pipe = get_pipeline("data_analysis_pipeline")
        data_splitter_output = pipe.runs[0].get_step(step="data_splitter")
        FacetStatisticsVisualizer().visualize(data_splitter_output)
    except Exception as e:
        logging.error(e)
