from whylogs import DatasetProfile  # type: ignore
from zenml.integrations.facets.visualizers.facet_statistics_visualizer import (
    FacetStatisticsVisualizer,
)
from zenml.integrations.whylogs.steps import whylogs_profiler_step
from zenml.integrations.whylogs.visualizers import WhylogsVisualizer
from zenml.integrations.whylogs.whylogs_step_decorator import enable_whylogs
from zenml.steps import step
from zenml.steps.step_context import StepContext


@step
def visualize_statistics(
    context: StepContext,
):
    pipe = context.metadata_store.get_pipeline("data_analysis_pipeline")
    ingest_data_outputs = pipe.runs[-1].get_step(name="ingest_data")
    FacetStatisticsVisualizer().visualize(ingest_data_outputs)


@step
def visualize_whylabs_statistics(context: StepContext):
    pipe = context.metadata_store.get_pipeline("data_analysis_pipeline")
    whylogs_train_outputs = pipe.runs[-1].get_step(name="train_data_profile")
    whylogs_test_outputs = pipe.runs[-1].get_step(name="test_data_profile")
    WhylogsVisualizer().visualize(whylogs_train_outputs)
    WhylogsVisualizer().visualize(whylogs_test_outputs)
