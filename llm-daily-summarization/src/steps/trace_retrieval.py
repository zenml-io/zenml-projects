"""
Trace retrieval step for fetching and visualizing Langfuse traces from the complete pipeline run.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated
from zenml import get_step_context, step
from zenml.logger import get_logger
from zenml.types import HTMLString

from ..utils.models import ProcessedData

logger = get_logger(__name__)


def _generate_langfuse_trace_url(trace_id: str, timestamp: str = None) -> str:
    """Generate direct Langfuse trace URL."""
    try:
        from ..utils.llm_config import get_langfuse_project_id

        langfuse_host = os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )
        project_id = get_langfuse_project_id()

        if not project_id:
            logger.warning("Could not determine project ID")
            return f"{langfuse_host}/traces/{trace_id}"

        # Create direct trace URL
        base_url = f"{langfuse_host}/project/{project_id}/traces/{trace_id}"
        if timestamp:
            return f"{base_url}?timestamp={timestamp}&display=details"
        return f"{base_url}?display=details"

    except Exception as e:
        logger.warning(f"Could not generate Langfuse trace URL: {e}")
        return f"{os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}/traces/{trace_id}"


@step
def retrieve_traces_step(
    processed_data: ProcessedData, time_window_minutes: int = 30
) -> Annotated[HTMLString, "traces_visualization"]:
    """
    Retrieve and visualize Langfuse traces from the specific pipeline session.

    Args:
        processed_data: Processed data containing trace IDs and session metadata
        time_window_minutes: Fallback time window in minutes if session filtering fails

    Returns:
        HTMLString: Comprehensive traces visualization
    """
    # Get run ID for tag-based filtering
    try:
        step_context = get_step_context()
        run_id = str(step_context.pipeline_run.id)
        pipeline_name = step_context.pipeline.name
        run_name = step_context.pipeline_run.name
    except Exception:
        import uuid

        run_id = str(uuid.uuid4())
        pipeline_name = "unknown_pipeline"
        run_name = "unknown_run"

    logger.info(f"Starting Langfuse trace retrieval for run ID: {run_id}")

    try:
        # Initialize Langfuse client
        from langfuse import Langfuse

        langfuse = Langfuse()

        # Fetch traces by tag filtering for the specific run ID
        logger.info(f"Fetching traces with tag: run_id:{run_id}")

        # Initialize time window variables for fallback
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        retrieval_method = "tag"

        # Try to fetch traces with the run_id as session_id (more precise)
        traces_response = langfuse.api.trace.list(
            session_id=run_id,
            limit=20,  # Limit to pipeline traces only
        )

        # If no traces found by session_id, try user_id matching
        if not traces_response.data:
            logger.info(
                f"No traces found with session_id {run_id}, trying user_id filter"
            )
            traces_response = langfuse.api.trace.list(
                user_id="zenml_pipeline",
                from_timestamp=start_time,
                to_timestamp=end_time,
                limit=20,
            )

        # Final fallback: look for traces with the run_id in the name or metadata
        if not traces_response.data:
            logger.warning(
                f"No traces found with session_id or user_id filters, falling back to time-based search"
            )
            retrieval_method = "time_window"

            logger.info(f"Fetching traces from {start_time} to {end_time}")

            traces_response = langfuse.api.trace.list(
                from_timestamp=start_time, to_timestamp=end_time, limit=20
            )

            # Filter traces to only those related to our pipeline
            if traces_response.data:
                pipeline_traces = []
                for trace in traces_response.data:
                    # Check if trace is related to our pipeline run
                    if (
                        trace.session_id == run_id
                        or trace.user_id == "zenml_pipeline"
                        or (
                            hasattr(trace, "metadata")
                            and trace.metadata
                            and run_id in str(trace.metadata)
                        )
                    ):
                        pipeline_traces.append(trace)

                # Create a mock response with filtered traces
                class MockResponse:
                    def __init__(self, data):
                        self.data = data

                traces_response = MockResponse(pipeline_traces)
                logger.info(
                    f"Filtered to {len(pipeline_traces)} pipeline-related traces"
                )
        else:
            logger.info(
                f"Found {len(traces_response.data)} traces with session_id {run_id}"
            )

        traces_data = []
        observations_data = []

        # Process each trace with rate limiting
        rate_limit_hit = False
        max_observations_per_trace = 5  # Only get essential observations

        for i, trace in enumerate(
            traces_response.data[:10]
        ):  # Limit to first 10 traces
            trace_dict = {
                "id": trace.id,
                "name": trace.name,
                "timestamp": trace.timestamp.isoformat()
                if trace.timestamp
                else None,
                "user_id": getattr(trace, "user_id", None),
                "session_id": getattr(trace, "session_id", None),
                "tags": getattr(trace, "tags", []),
                "metadata": getattr(trace, "metadata", {}),
                "input": getattr(trace, "input", None),
                "output": getattr(trace, "output", None),
                "level": getattr(trace, "level", "DEFAULT"),
                "status_message": getattr(trace, "status_message", None),
                "version": getattr(trace, "version", None),
            }
            traces_data.append(trace_dict)

            # Skip observation fetching if we've hit rate limits or processed enough traces
            if (
                rate_limit_hit or i >= 5
            ):  # Limit to first 5 traces to reduce API calls
                continue

            # Fetch observations for this trace with retry and rate limiting
            try:
                observations_response = _fetch_observations_with_retry(
                    langfuse, trace.id, max_observations_per_trace
                )

                if observations_response is None:
                    rate_limit_hit = True
                    logger.warning(
                        f"Rate limit hit, skipping remaining observation fetches"
                    )
                    continue

                for obs in observations_response.data:
                    # Filter to only show relevant observations (generations and important spans)
                    obs_type = getattr(obs, "type", "unknown")
                    obs_name = getattr(obs, "name", "Unnamed Observation")

                    # Only include observations that are:
                    # 1. Generations (LLM calls)
                    # 2. Observations from our agents (contain "summarize" or "extract" or "agent")
                    if obs_type.upper() == "GENERATION" or any(
                        keyword in obs_name.lower()
                        for keyword in [
                            "summarize",
                            "extract",
                            "agent",
                            "conversation",
                            "task",
                        ]
                    ):
                        obs_dict = {
                            "id": obs.id,
                            "trace_id": trace.id,
                            "name": obs_name,
                            "type": obs_type,
                            "start_time": obs.start_time.isoformat()
                            if getattr(obs, "start_time", None)
                            else None,
                            "end_time": obs.end_time.isoformat()
                            if getattr(obs, "end_time", None)
                            else None,
                            "completion_start_time": obs.completion_start_time.isoformat()
                            if getattr(obs, "completion_start_time", None)
                            else None,
                            "model": getattr(obs, "model", None),
                            "input": getattr(obs, "input", None),
                            "output": getattr(obs, "output", None),
                            "usage": _serialize_usage(
                                getattr(obs, "usage", None)
                            ),
                            "level": getattr(obs, "level", "DEFAULT"),
                            "status_message": getattr(
                                obs, "status_message", None
                            ),
                            "parent_observation_id": getattr(
                                obs, "parent_observation_id", None
                            ),
                            "metadata": getattr(obs, "metadata", {}),
                            "version": getattr(obs, "version", None),
                        }
                        observations_data.append(obs_dict)

                # Add small delay between requests to avoid rate limits
                if (
                    i < len(traces_response.data) - 1
                ):  # Don't sleep after the last request
                    time.sleep(0.1)  # 100ms delay

            except Exception as e:
                logger.warning(
                    f"Failed to fetch observations for trace {trace.id}: {e}"
                )
                if "rate limit" in str(e).lower():
                    rate_limit_hit = True

        logger.info(
            f"Retrieved {len(traces_data)} traces and {len(observations_data)} observations using {retrieval_method} method"
        )

        # Compile all data
        all_traces_data = {
            "traces": traces_data,
            "observations": observations_data,
            "pipeline_trace_id": processed_data.agent_trace_id,
            "run_id": processed_data.run_id,
            "retrieval_timestamp": datetime.utcnow().isoformat(),
            "retrieval_method": retrieval_method,
            "run_metadata": {
                "run_id": run_id,
                "pipeline_name": pipeline_name,
                "run_name": run_name,
                "langfuse_url": _generate_langfuse_trace_url(run_id),
            },
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "minutes": time_window_minutes,
            },
            "summary": {
                "total_traces": len(traces_data),
                "total_observations": len(observations_data),
                "trace_types": _analyze_trace_types(traces_data),
                "observation_types": _analyze_observation_types(
                    observations_data
                ),
            },
        }

        # Create visualization
        traces_viz = _create_traces_visualization(
            all_traces_data, processed_data
        )

        logger.info("Trace retrieval and visualization complete")
        return traces_viz

    except Exception as e:
        logger.error(f"Error retrieving traces: {e}")

        # Return minimal data with error information
        error_data = {
            "traces": [],
            "observations": [],
            "pipeline_trace_id": processed_data.agent_trace_id,
            "retrieval_timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "summary": {
                "total_traces": 0,
                "total_observations": 0,
                "trace_types": {},
                "observation_types": {},
            },
        }

        error_viz = _create_error_visualization(error_data)
        return error_viz


def _fetch_observations_with_retry(
    langfuse, trace_id: str, limit: int = 10, max_retries: int = 2
) -> Optional[Any]:
    """Fetch observations with retry logic to handle rate limits."""

    for attempt in range(max_retries + 1):
        try:
            return langfuse.api.observations.get_many(
                trace_id=trace_id, limit=limit
            )
        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a rate limit error
            if "rate limit" in error_msg or "429" in error_msg:
                if attempt < max_retries:
                    wait_time = (
                        2**attempt
                    ) * 0.5  # Exponential backoff: 0.5s, 1s, 2s
                    logger.info(
                        f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(
                        f"Rate limit hit, giving up after {max_retries} retries"
                    )
                    return None
            else:
                # Non-rate-limit error, propagate it
                raise e

    return None


def _serialize_usage(usage_obj: Any) -> Optional[Dict[str, Any]]:
    """Convert Langfuse Usage object to a serializable dictionary."""
    if usage_obj is None:
        return None

    try:
        # Try to convert to dict if it has the expected attributes
        if hasattr(usage_obj, "__dict__"):
            usage_dict = {}
            for attr in [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "total",
                "total_cost",
            ]:
                if hasattr(usage_obj, attr):
                    value = getattr(usage_obj, attr)
                    # Convert to basic types
                    if isinstance(value, (int, float, str)):
                        usage_dict[attr] = value
                    elif value is not None:
                        usage_dict[attr] = str(value)
            return usage_dict if usage_dict else None
        elif isinstance(usage_obj, dict):
            # Already a dict
            return usage_obj
        else:
            # Try to convert to string as fallback
            return {"raw_usage": str(usage_obj)}
    except Exception:
        return None


def _analyze_trace_types(traces_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze trace types and count occurrences."""
    types_count = {}
    for trace in traces_data:
        trace_name = trace.get("name", "unknown")
        types_count[trace_name] = types_count.get(trace_name, 0) + 1
    return types_count


def _analyze_observation_types(
    observations_data: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Analyze observation types and count occurrences."""
    types_count = {}
    for obs in observations_data:
        obs_type = obs.get("type", "unknown")
        types_count[obs_type] = types_count.get(obs_type, 0) + 1
    return types_count


def _create_traces_visualization(
    traces_data: Dict[str, Any], processed_data: ProcessedData
) -> HTMLString:
    """Create modern, interactive HTML visualization for Langfuse traces."""

    traces = traces_data["traces"]
    observations = traces_data["observations"]
    run_metadata = traces_data.get("run_metadata", {})
    retrieval_method = traces_data.get("retrieval_method", "unknown")

    # Group observations by trace for better organization
    trace_obs_map = {}
    for obs in observations:
        trace_id = obs.get("trace_id")
        if trace_id not in trace_obs_map:
            trace_obs_map[trace_id] = []
        trace_obs_map[trace_id].append(obs)

    # Calculate aggregate statistics
    total_tokens = sum(
        obs.get("usage", {}).get("total", 0)
        if isinstance(obs.get("usage"), dict)
        else 0
        for obs in observations
    )
    total_cost = sum(
        obs.get("usage", {}).get("total_cost", 0.0)
        if isinstance(obs.get("usage"), dict)
        else 0.0
        for obs in observations
    )
    avg_duration = _calculate_avg_duration(observations)
    generation_count = len(
        [obs for obs in observations if obs.get("type") == "GENERATION"]
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔍 Langfuse Traces - Interactive Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: #0a0a0a;
                color: #ffffff;
                line-height: 1.6;
                overflow-x: hidden;
            }}
            
            .dashboard {{
                min-height: 100vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                position: relative;
            }}
            
            .dashboard::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: 
                    radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                    radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
                pointer-events: none;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
                position: relative;
                z-index: 1;
            }}
            
            .header {{
                text-align: center;
                margin-bottom: 3rem;
                padding: 3rem 2rem;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            .header h1 {{
                font-size: 3.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1rem;
            }}
            
            .header-meta {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 2rem;
                font-size: 0.9rem;
                opacity: 0.9;
            }}
            
            .meta-item {{
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 2rem;
                margin-bottom: 3rem;
            }}
            
            .stat-card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .stat-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
                background-size: 200% 100%;
                animation: shimmer 3s ease-in-out infinite;
            }}
            
            @keyframes shimmer {{
                0%, 100% {{ background-position: 200% 0; }}
                50% {{ background-position: -200% 0; }}
            }}
            
            .stat-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
            }}
            
            .stat-icon {{
                font-size: 3rem;
                margin-bottom: 1rem;
                opacity: 0.8;
            }}
            
            .stat-value {{
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .stat-label {{
                font-size: 1.1rem;
                opacity: 0.8;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: 500;
            }}
            
            .traces-section {{
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                padding: 3rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            .section-title {{
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 2rem;
                background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .trace-tree {{
                position: relative;
            }}
            
            .trace-node {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                margin-bottom: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.2);
                overflow: hidden;
                box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }}
            
            .trace-node:hover {{
                transform: translateX(8px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            .trace-header {{
                padding: 2rem;
                background: rgba(255, 255, 255, 0.05);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .trace-title {{
                font-size: 1.4rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            
            .trace-badge {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .trace-meta {{
                display: flex;
                gap: 2rem;
                font-size: 0.9rem;
                opacity: 0.8;
            }}
            
            .trace-content {{
                padding: 0;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease, padding 0.3s ease;
            }}
            
            .trace-content.expanded {{
                max-height: 2000px;
                padding: 2rem;
            }}
            
            .observations-grid {{
                display: grid;
                gap: 1.5rem;
            }}
            
            .observation {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 1.5rem;
                border-left: 4px solid;
                position: relative;
            }}
            
            .observation.generation {{
                border-left-color: #10b981;
            }}
            
            .observation.span {{
                border-left-color: #3b82f6;
            }}
            
            .observation.event {{
                border-left-color: #f59e0b;
            }}
            
            .obs-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }}
            
            .obs-name {{
                font-weight: 600;
                font-size: 1.1rem;
            }}
            
            .obs-type {{
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .obs-type.generation {{
                background: rgba(16, 185, 129, 0.2);
                color: #10b981;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }}
            
            .obs-type.span {{
                background: rgba(59, 130, 246, 0.2);
                color: #3b82f6;
                border: 1px solid rgba(59, 130, 246, 0.3);
            }}
            
            .obs-type.event {{
                background: rgba(245, 158, 11, 0.2);
                color: #f59e0b;
                border: 1px solid rgba(245, 158, 11, 0.3);
            }}
            
            .obs-details {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }}
            
            .obs-detail {{
                background: rgba(255, 255, 255, 0.05);
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .detail-label {{
                font-size: 0.8rem;
                opacity: 0.7;
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .detail-value {{
                font-weight: 600;
                font-size: 1rem;
            }}
            
            .usage-stats {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 1rem;
                margin-top: 1rem;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
            }}
            
            .usage-stat {{
                text-align: center;
            }}
            
            .usage-value {{
                font-size: 1.5rem;
                font-weight: 700;
                color: #10b981;
            }}
            
            .usage-label {{
                font-size: 0.75rem;
                opacity: 0.7;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .expand-btn {{
                background: none;
                border: none;
                color: rgba(255, 255, 255, 0.7);
                font-size: 1.5rem;
                cursor: pointer;
                transition: transform 0.3s ease, color 0.3s ease;
            }}
            
            .expand-btn.expanded {{
                transform: rotate(180deg);
                color: #ffffff;
            }}
            
            .empty-state {{
                text-align: center;
                padding: 4rem 2rem;
                opacity: 0.7;
            }}
            
            .empty-state h3 {{
                font-size: 2rem;
                margin-bottom: 1rem;
            }}
            
            .langfuse-link {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                text-decoration: none;
                padding: 1rem 2rem;
                border-radius: 50px;
                font-weight: 500;
                transition: all 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.2);
                margin-top: 1rem;
            }}
            
            .langfuse-link:hover {{
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    padding: 1rem;
                }}
                
                .header h1 {{
                    font-size: 2.5rem;
                }}
                
                .stats-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .header-meta {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        <script>
            function toggleTrace(traceId) {{
                const content = document.getElementById('trace-content-' + traceId);
                const btn = document.getElementById('expand-btn-' + traceId);
                
                if (content.classList.contains('expanded')) {{
                    content.classList.remove('expanded');
                    btn.classList.remove('expanded');
                }} else {{
                    content.classList.add('expanded');
                    btn.classList.add('expanded');
                }}
            }}
        </script>
    </head>
    <body>
        <div class="dashboard">
            <div class="container">
                <div class="header">
                    <h1>🔍 Langfuse Traces</h1>
                    <p style="font-size: 1.2rem; opacity: 0.9; margin-bottom: 2rem;">
                        Real-time observability for your LLM pipeline execution
                    </p>
                    
                    <div class="header-meta">
                        <div class="meta-item">
                            <strong>Pipeline:</strong> {run_metadata.get('pipeline_name', 'Unknown')}
                        </div>
                        <div class="meta-item">
                            <strong>Run:</strong> {run_metadata.get('run_name', 'Unknown')}
                        </div>
                        <div class="meta-item">
                            <strong>Method:</strong> {retrieval_method.title()}
                        </div>
                        <div class="meta-item">
                            <strong>Retrieved:</strong> {datetime.fromisoformat(traces_data['retrieval_timestamp']).strftime('%H:%M:%S UTC')}
                        </div>
                    </div>
                    
                    {f'<a href="{run_metadata.get("langfuse_url", "#")}" target="_blank" class="langfuse-link">🔗 View in Langfuse Dashboard</a>' if run_metadata.get("langfuse_url") else ""}
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">📊</div>
                        <div class="stat-value">{len(traces)}</div>
                        <div class="stat-label">Total Traces</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🔬</div>
                        <div class="stat-value">{len(observations)}</div>
                        <div class="stat-label">Observations</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">🤖</div>
                        <div class="stat-value">{generation_count}</div>
                        <div class="stat-label">Generations</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">⚡</div>
                        <div class="stat-value">{total_tokens:,}</div>
                        <div class="stat-label">Total Tokens</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">💰</div>
                        <div class="stat-value">${total_cost:.4f}</div>
                        <div class="stat-label">Estimated Cost</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-icon">⏱️</div>
                        <div class="stat-value">{avg_duration}</div>
                        <div class="stat-label">Avg Duration</div>
                    </div>
                </div>
    """

    if traces:
        html_content += f"""
                <div class="traces-section">
                    <h2 class="section-title">📋 Trace Execution Flow</h2>
                    <div class="trace-tree">
        """

        for i, trace in enumerate(traces):
            trace_observations = trace_obs_map.get(trace["id"], [])
            trace_id_short = trace["id"][:8]

            # Sort observations by start time if available
            trace_observations.sort(
                key=lambda x: x.get("start_time", ""), reverse=False
            )

            html_content += f"""
                        <div class="trace-node">
                            <div class="trace-header" onclick="toggleTrace('{trace_id_short}')">
                                <div class="trace-title">
                                    <span style="font-size: 1.5rem;">🎯</span>
                                    <span>{trace.get('name', f'Trace {i+1}')}</span>
                                    <div class="trace-badge">{len(trace_observations)} observations</div>
                                </div>
                                <div class="trace-meta">
                                    <span>ID: {trace_id_short}...</span>
                                    <span>{datetime.fromisoformat(trace['timestamp']).strftime('%H:%M:%S') if trace.get('timestamp') else 'No timestamp'}</span>
                                    <button class="expand-btn" id="expand-btn-{trace_id_short}">▼</button>
                                </div>
                            </div>
                            
                            <div class="trace-content" id="trace-content-{trace_id_short}">
                                <div class="observations-grid">
            """

            for obs in trace_observations:
                obs_type = obs.get("type", "unknown").lower()
                obs_name = obs.get("name", "Unnamed Operation")
                model = obs.get("model", "Not specified")
                duration = _calculate_duration(
                    obs.get("start_time"), obs.get("end_time")
                )

                # Get usage info
                usage = obs.get("usage", {})
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get(
                        "total", prompt_tokens + completion_tokens
                    )
                    cost = usage.get("total_cost", 0.0)
                else:
                    prompt_tokens = completion_tokens = total_tokens = 0
                    cost = 0.0

                html_content += f"""
                                    <div class="observation {obs_type}">
                                        <div class="obs-header">
                                            <div class="obs-name">{'🤖' if obs_type == 'generation' else '📝' if obs_type == 'span' else '⚡'} {obs_name}</div>
                                            <div class="obs-type {obs_type}">{obs_type}</div>
                                        </div>
                                        
                                        <div class="obs-details">
                                            <div class="obs-detail">
                                                <div class="detail-label">Model</div>
                                                <div class="detail-value">{model}</div>
                                            </div>
                                            <div class="obs-detail">
                                                <div class="detail-label">Duration</div>
                                                <div class="detail-value">{duration}</div>
                                            </div>
                                            <div class="obs-detail">
                                                <div class="detail-label">Status</div>
                                                <div class="detail-value">{obs.get('level', 'DEFAULT')}</div>
                                            </div>
                                            <div class="obs-detail">
                                                <div class="detail-label">Parent</div>
                                                <div class="detail-value">{'Yes' if obs.get('parent_observation_id') else 'Root'}</div>
                                            </div>
                                        </div>
                """

                if total_tokens > 0 or cost > 0:
                    html_content += f"""
                                        <div class="usage-stats">
                                            <div class="usage-stat">
                                                <div class="usage-value">{prompt_tokens:,}</div>
                                                <div class="usage-label">Input Tokens</div>
                                            </div>
                                            <div class="usage-stat">
                                                <div class="usage-value">{completion_tokens:,}</div>
                                                <div class="usage-label">Output Tokens</div>
                                            </div>
                                            <div class="usage-stat">
                                                <div class="usage-value">{total_tokens:,}</div>
                                                <div class="usage-label">Total Tokens</div>
                                            </div>
                                            <div class="usage-stat">
                                                <div class="usage-value">${cost:.4f}</div>
                                                <div class="usage-label">Cost</div>
                                            </div>
                                        </div>
                    """

                html_content += "</div>"

            html_content += """
                                </div>
                            </div>
                        </div>
            """

        html_content += """
                    </div>
                </div>
        """
    else:
        html_content += """
                <div class="traces-section">
                    <div class="empty-state">
                        <h3>🔍 No traces found</h3>
                        <p>No traces were found for this pipeline run. This could mean:</p>
                        <ul style="text-align: left; display: inline-block; margin-top: 1rem;">
                            <li>The pipeline hasn't started LLM calls yet</li>
                            <li>Langfuse integration is not properly configured</li>
                            <li>The time window is too narrow</li>
                        </ul>
                    </div>
                </div>
        """

    html_content += """
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html_content)


def _create_error_visualization(error_data: Dict[str, Any]) -> HTMLString:
    """Create error visualization when trace retrieval fails."""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Langfuse Traces - Error</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }}
            .error-container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .error-icon {{
                font-size: 4em;
                color: #dc3545;
                margin-bottom: 20px;
            }}
            .error-title {{
                color: #dc3545;
                font-size: 2em;
                margin-bottom: 20px;
            }}
            .error-message {{
                color: #666;
                font-size: 1.1em;
                line-height: 1.6;
                margin-bottom: 30px;
            }}
            .error-details {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: left;
                font-family: monospace;
                color: #dc3545;
            }}
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">⚠️</div>
            <h1 class="error-title">Trace Retrieval Failed</h1>
            <div class="error-message">
                Unable to retrieve traces from Langfuse. This could be due to:
                <ul style="text-align: left; display: inline-block;">
                    <li>Missing or incorrect Langfuse credentials</li>
                    <li>Network connectivity issues</li>
                    <li>No traces available in the specified time window</li>
                    <li>Langfuse service unavailability</li>
                </ul>
            </div>
            <div class="error-details">
                <strong>Error:</strong> {error_data.get('error', 'Unknown error')}<br>
                <strong>Pipeline Trace ID:</strong> {error_data.get('pipeline_trace_id', 'Unknown')}<br>
                <strong>Timestamp:</strong> {error_data.get('retrieval_timestamp', 'Unknown')}
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html_content)


def _calculate_duration(
    start_time: Optional[str], end_time: Optional[str]
) -> str:
    """Calculate duration between start and end times."""
    if not start_time or not end_time:
        return "Unknown"

    try:
        start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        duration = end - start

        total_seconds = duration.total_seconds()
        if total_seconds < 1:
            return f"{total_seconds*1000:.0f}ms"
        else:
            return f"{total_seconds:.2f}s"
    except Exception:
        return "Unknown"


def _calculate_avg_duration(observations: List[Dict[str, Any]]) -> str:
    """Calculate average duration across all observations."""
    durations = []
    for obs in observations:
        start_time = obs.get("start_time")
        end_time = obs.get("end_time")
        if start_time and end_time:
            try:
                start = datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")
                )
                end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                duration = (end - start).total_seconds()
                durations.append(duration)
            except Exception:
                continue

    if not durations:
        return "Unknown"

    avg_seconds = sum(durations) / len(durations)
    if avg_seconds < 1:
        return f"{avg_seconds*1000:.0f}ms"
    else:
        return f"{avg_seconds:.2f}s"
