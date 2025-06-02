"""Step to collect tracing metadata from Langfuse for the pipeline run."""

import logging
from typing import Annotated, Dict

from materializers.tracing_metadata_materializer import (
    TracingMetadataMaterializer,
)
from utils.pydantic_models import (
    PromptTypeMetrics,
    QueryContext,
    SearchData,
    TracingMetadata,
)
from utils.tracing_metadata_utils import (
    get_observations_for_trace,
    get_prompt_type_statistics,
    get_trace_stats,
    get_traces_by_name,
)
from zenml import get_step_context, step

logger = logging.getLogger(__name__)


@step(
    enable_cache=False,
    output_materializers={
        "tracing_metadata": TracingMetadataMaterializer,
    },
)
def collect_tracing_metadata_step(
    query_context: QueryContext,
    search_data: SearchData,
    langfuse_project_name: str,
) -> Annotated[TracingMetadata, "tracing_metadata"]:
    """Collect tracing metadata from Langfuse for the current pipeline run.

    This step gathers comprehensive metrics about token usage, costs, and performance
    for the entire pipeline run, providing insights into resource consumption.

    Args:
        query_context: The query context (for reference)
        search_data: The search data containing cost information
        langfuse_project_name: Langfuse project name for accessing traces

    Returns:
        TracingMetadata with comprehensive cost and performance metrics
    """
    ctx = get_step_context()
    pipeline_run_name = ctx.pipeline_run.name
    pipeline_run_id = str(ctx.pipeline_run.id)

    logger.info(
        f"Collecting tracing metadata for pipeline run: {pipeline_run_name} (ID: {pipeline_run_id})"
    )

    # Initialize the metadata object
    metadata = TracingMetadata(
        pipeline_run_name=pipeline_run_name,
        pipeline_run_id=pipeline_run_id,
        trace_name=pipeline_run_name,
        trace_id=pipeline_run_id,
    )

    try:
        # Fetch the trace for this pipeline run
        # The trace_name is the pipeline run name
        traces = get_traces_by_name(name=pipeline_run_name, limit=1)

        if not traces:
            logger.warning(
                f"No trace found for pipeline run: {pipeline_run_name}"
            )
            # Still add search costs before returning
            _add_search_costs_to_metadata(metadata, search_data)
            return metadata

        trace = traces[0]

        # Get comprehensive trace stats
        trace_stats = get_trace_stats(trace)

        # Update metadata with trace stats
        metadata.trace_id = trace.id
        metadata.total_cost = trace_stats["total_cost"]
        metadata.total_input_tokens = trace_stats["input_tokens"]
        metadata.total_output_tokens = trace_stats["output_tokens"]
        metadata.total_tokens = (
            trace_stats["input_tokens"] + trace_stats["output_tokens"]
        )
        metadata.total_latency_seconds = trace_stats["latency_seconds"]
        metadata.formatted_latency = trace_stats["latency_formatted"]
        metadata.observation_count = trace_stats["observation_count"]
        metadata.models_used = trace_stats["models_used"]
        metadata.trace_tags = trace_stats.get("tags", [])
        metadata.trace_metadata = trace_stats.get("metadata", {})

        # Get model-specific breakdown
        observations = get_observations_for_trace(trace_id=trace.id)
        model_costs = {}
        model_tokens = {}
        step_costs = {}
        step_tokens = {}

        for obs in observations:
            if obs.model:
                # Track by model
                if obs.model not in model_costs:
                    model_costs[obs.model] = 0.0
                    model_tokens[obs.model] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }

                if obs.calculated_total_cost:
                    model_costs[obs.model] += obs.calculated_total_cost

                if obs.usage:
                    input_tokens = obs.usage.input or 0
                    output_tokens = obs.usage.output or 0
                    model_tokens[obs.model]["input_tokens"] += input_tokens
                    model_tokens[obs.model]["output_tokens"] += output_tokens
                    model_tokens[obs.model]["total_tokens"] += (
                        input_tokens + output_tokens
                    )

            # Track by step (using observation name as step indicator)
            if obs.name:
                step_name = obs.name

                if step_name not in step_costs:
                    step_costs[step_name] = 0.0
                    step_tokens[step_name] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }

                if obs.calculated_total_cost:
                    step_costs[step_name] += obs.calculated_total_cost

                if obs.usage:
                    input_tokens = obs.usage.input or 0
                    output_tokens = obs.usage.output or 0
                    step_tokens[step_name]["input_tokens"] += input_tokens
                    step_tokens[step_name]["output_tokens"] += output_tokens

        metadata.cost_breakdown_by_model = model_costs
        metadata.model_token_breakdown = model_tokens
        metadata.step_costs = step_costs
        metadata.step_tokens = step_tokens

        # Collect prompt-level metrics
        try:
            prompt_stats = get_prompt_type_statistics(trace_id=trace.id)

            # Convert to PromptTypeMetrics objects
            prompt_metrics_list = []
            for prompt_type, stats in prompt_stats.items():
                prompt_metrics = PromptTypeMetrics(
                    prompt_type=prompt_type,
                    total_cost=stats["cost"],
                    input_tokens=stats["input_tokens"],
                    output_tokens=stats["output_tokens"],
                    call_count=stats["count"],
                    avg_cost_per_call=stats["avg_cost_per_call"],
                    percentage_of_total_cost=stats["percentage_of_total_cost"],
                )
                prompt_metrics_list.append(prompt_metrics)

            # Sort by total cost descending
            prompt_metrics_list.sort(key=lambda x: x.total_cost, reverse=True)
            metadata.prompt_metrics = prompt_metrics_list

            logger.info(
                f"Collected prompt-level metrics for {len(prompt_metrics_list)} prompt types"
            )
        except Exception as e:
            logger.warning(f"Failed to collect prompt-level metrics: {str(e)}")

        # Add search costs from the SearchData artifact
        _add_search_costs_to_metadata(metadata, search_data)

        total_search_cost = sum(metadata.search_costs.values())
        logger.info(
            f"Successfully collected tracing metadata - "
            f"LLM Cost: ${metadata.total_cost:.4f}, "
            f"Search Cost: ${total_search_cost:.4f}, "
            f"Total Cost: ${metadata.total_cost + total_search_cost:.4f}, "
            f"Tokens: {metadata.total_tokens:,}, "
            f"Models: {metadata.models_used}, "
            f"Duration: {metadata.formatted_latency}"
        )

    except Exception as e:
        logger.error(
            f"Failed to collect tracing metadata for pipeline run {pipeline_run_name}: {str(e)}"
        )
        # Return metadata with whatever we could collect
        # Still try to get search costs even if Langfuse failed
        _add_search_costs_to_metadata(metadata, search_data)

    # Add tags to the artifact
    # add_tags(
    #     tags=["exa", "tavily", "llm", "cost", "tracing"],
    #     artifact_name="tracing_metadata",
    #     infer_artifact=True,
    # )

    return metadata


def _add_search_costs_to_metadata(
    metadata: TracingMetadata, search_data: SearchData
) -> None:
    """Add search costs from SearchData to TracingMetadata.

    Args:
        metadata: The TracingMetadata object to update
        search_data: The SearchData containing cost information
    """
    if search_data.search_costs:
        metadata.search_costs = search_data.search_costs.copy()
        logger.info(f"Added search costs: {metadata.search_costs}")

    if search_data.search_cost_details:
        # Convert SearchCostDetail objects to dicts for backward compatibility
        metadata.search_cost_details = [
            {
                "provider": detail.provider,
                "query": detail.query,
                "cost": detail.cost,
                "timestamp": detail.timestamp,
                "step": detail.step,
                "sub_question": detail.sub_question,
            }
            for detail in search_data.search_cost_details
        ]

        # Count queries by provider
        search_queries_count: Dict[str, int] = {}
        for detail in search_data.search_cost_details:
            provider = detail.provider
            search_queries_count[provider] = (
                search_queries_count.get(provider, 0) + 1
            )
        metadata.search_queries_count = search_queries_count

        logger.info(
            f"Added {len(metadata.search_cost_details)} search cost detail entries"
        )
