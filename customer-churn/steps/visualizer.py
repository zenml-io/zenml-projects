import logging

from zenml.integrations.facets.visualizers.facet_statistics_visualizer import (
    FacetStatisticsVisualizer,
)
from zenml.repository import Repository


def visualize_statistics():
    """
    This function will get the last pipeline run, get the last step of the pipeline run, and then
    visualize the statistics of the last step.
    """
    try:
        repo = Repository()
        pipe = repo.get_pipeline(pipeline_name="data_analysis_pipeline")
        ingest_data = pipe.runs[-1].get_step(name="ingest_data")
        FacetStatisticsVisualizer().visualize(ingest_data)
    except Exception as e:
        logging.error(e)


def visualize_train_test_statistics():
    """
    It visualizes the statistics of the train and test datasets.
    """
    try:
        repo = Repository()
        pipe = repo.get_pipeline(pipeline_name="data_analysis_pipeline")
        data_splitter_output = pipe.runs[-1].get_step(name="data_splitter")
        FacetStatisticsVisualizer().visualize(data_splitter_output)
    except Exception as e:
        logging.error(e)
