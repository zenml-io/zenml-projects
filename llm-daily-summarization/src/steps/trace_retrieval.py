"""
Trace retrieval step for fetching and visualizing Langfuse traces from the complete pipeline run.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

from langfuse import get_client
from zenml import step
from zenml.logger import get_logger
from zenml.types import HTMLString

from ..utils.models import ProcessedData
from ..utils.session_manager import get_session_manager

logger = get_logger(__name__)


@step
def retrieve_traces_step(
    processed_data: ProcessedData,
    time_window_minutes: int = 30
) -> Annotated[HTMLString, "traces_visualization"]:
    """
    Retrieve and visualize Langfuse traces from the specific pipeline session.
    
    Args:
        processed_data: Processed data containing trace IDs and session metadata
        time_window_minutes: Fallback time window in minutes if session filtering fails
        
    Returns:
        HTMLString: Comprehensive traces visualization
    """
    logger.info(f"Starting Langfuse trace retrieval for session: {processed_data.session_id}")
    
    # Get session manager for additional metadata
    session_manager = get_session_manager()
    session_metadata = session_manager.get_session_metadata()
    
    try:
        # Initialize Langfuse client
        langfuse = get_client()
        
        # First try to fetch traces by session ID
        logger.info(f"Fetching traces for session ID: {processed_data.session_id}")
        
        # Initialize time window variables (used for metadata regardless of retrieval method)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        retrieval_method = "session"
        
        # Fetch traces for the specific session
        traces_response = langfuse.api.trace.list(
            session_id=processed_data.session_id,
            limit=100  # Should be sufficient for a single pipeline run
        )
        
        # If no traces found by session, fallback to time-based search
        if not traces_response.data:
            logger.warning(f"No traces found for session {processed_data.session_id}, falling back to time-based search")
            retrieval_method = "time_window"
            
            logger.info(f"Fetching traces from {start_time} to {end_time}")
            
            traces_response = langfuse.api.trace.list(
                from_timestamp=start_time,
                to_timestamp=end_time,
                limit=100
            )
        else:
            logger.info(f"Found {len(traces_response.data)} traces for session {processed_data.session_id}")
        
        traces_data = []
        observations_data = []
        
        # Process each trace with rate limiting
        rate_limit_hit = False
        max_observations_per_trace = 10  # Reduce to limit API calls
        
        for i, trace in enumerate(traces_response.data):
            trace_dict = {
                "id": trace.id,
                "name": trace.name,
                "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
                "user_id": getattr(trace, 'user_id', None),
                "session_id": getattr(trace, 'session_id', None),
                "tags": getattr(trace, 'tags', []),
                "metadata": getattr(trace, 'metadata', {}),
                "input": getattr(trace, 'input', None),
                "output": getattr(trace, 'output', None),
                "level": getattr(trace, 'level', 'DEFAULT'),
                "status_message": getattr(trace, 'status_message', None),
                "version": getattr(trace, 'version', None)
            }
            traces_data.append(trace_dict)
            
            # Skip observation fetching if we've hit rate limits or processed enough traces
            if rate_limit_hit or i >= 20:  # Limit to first 20 traces to reduce API calls
                continue
                
            # Fetch observations for this trace with retry and rate limiting
            try:
                observations_response = _fetch_observations_with_retry(
                    langfuse, trace.id, max_observations_per_trace
                )
                
                if observations_response is None:
                    rate_limit_hit = True
                    logger.warning(f"Rate limit hit, skipping remaining observation fetches")
                    continue
                
                for obs in observations_response.data:
                    obs_dict = {
                        "id": obs.id,
                        "trace_id": trace.id,
                        "name": getattr(obs, 'name', 'Unnamed Observation'),
                        "type": getattr(obs, 'type', 'unknown'),
                        "start_time": obs.start_time.isoformat() if getattr(obs, 'start_time', None) else None,
                        "end_time": obs.end_time.isoformat() if getattr(obs, 'end_time', None) else None,
                        "completion_start_time": obs.completion_start_time.isoformat() if getattr(obs, 'completion_start_time', None) else None,
                        "model": getattr(obs, 'model', None),
                        "input": getattr(obs, 'input', None),
                        "output": getattr(obs, 'output', None),
                        "usage": _serialize_usage(getattr(obs, 'usage', None)),
                        "level": getattr(obs, 'level', 'DEFAULT'),
                        "status_message": getattr(obs, 'status_message', None),
                        "parent_observation_id": getattr(obs, 'parent_observation_id', None),
                        "metadata": getattr(obs, 'metadata', {}),
                        "version": getattr(obs, 'version', None)
                    }
                    observations_data.append(obs_dict)
                    
                # Add small delay between requests to avoid rate limits
                if i < len(traces_response.data) - 1:  # Don't sleep after the last request
                    time.sleep(0.1)  # 100ms delay
                    
            except Exception as e:
                logger.warning(f"Failed to fetch observations for trace {trace.id}: {e}")
                if "rate limit" in str(e).lower():
                    rate_limit_hit = True
        
        logger.info(f"Retrieved {len(traces_data)} traces and {len(observations_data)} observations using {retrieval_method} method")
        
        # Compile all data
        all_traces_data = {
            "traces": traces_data,
            "observations": observations_data,
            "pipeline_trace_id": processed_data.agent_trace_id,
            "session_id": processed_data.session_id,
            "retrieval_timestamp": datetime.utcnow().isoformat(),
            "retrieval_method": retrieval_method,
            "session_metadata": session_metadata,
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "minutes": time_window_minutes
            },
            "summary": {
                "total_traces": len(traces_data),
                "total_observations": len(observations_data),
                "trace_types": _analyze_trace_types(traces_data),
                "observation_types": _analyze_observation_types(observations_data)
            }
        }
        
        # Create visualization
        traces_viz = _create_traces_visualization(all_traces_data, processed_data)
        
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
                "observation_types": {}
            }
        }
        
        error_viz = _create_error_visualization(error_data)
        return error_viz


def _fetch_observations_with_retry(langfuse, trace_id: str, limit: int = 10, max_retries: int = 2) -> Optional[Any]:
    """Fetch observations with retry logic to handle rate limits."""
    
    for attempt in range(max_retries + 1):
        try:
            return langfuse.api.observations.get_many(
                trace_id=trace_id,
                limit=limit
            )
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate limit" in error_msg or "429" in error_msg:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s
                    logger.info(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"Rate limit hit, giving up after {max_retries} retries")
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
        if hasattr(usage_obj, '__dict__'):
            usage_dict = {}
            for attr in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'total', 'total_cost']:
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


def _analyze_observation_types(observations_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze observation types and count occurrences."""
    types_count = {}
    for obs in observations_data:
        obs_type = obs.get("type", "unknown")
        types_count[obs_type] = types_count.get(obs_type, 0) + 1
    return types_count


def _create_traces_visualization(traces_data: Dict[str, Any], processed_data: ProcessedData) -> HTMLString:
    """Create comprehensive HTML visualization for Langfuse traces."""
    
    traces = traces_data["traces"]
    observations = traces_data["observations"]
    summary = traces_data["summary"]
    session_metadata = traces_data.get("session_metadata", {})
    retrieval_method = traces_data.get("retrieval_method", "unknown")
    
    # Calculate some statistics
    total_tokens = 0
    total_cost = 0.0
    
    for obs in observations:
        if obs.get("usage"):
            usage = obs["usage"]
            if isinstance(usage, dict):
                total_tokens += usage.get("total", 0)
                total_cost += usage.get("total_cost", 0.0)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Langfuse Traces Dashboard - Pipeline Run</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 2.5em;
                font-weight: 300;
            }}
            
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                padding: 30px;
                background: #f8f9fa;
            }}
            
            .summary-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}
            
            .summary-value {{
                font-size: 2.5em;
                font-weight: 700;
                color: #8b5cf6;
                margin: 10px 0;
            }}
            
            .summary-label {{
                color: #666;
                font-size: 1em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            .content-section {{
                padding: 30px;
            }}
            
            .section-title {{
                color: #2c3e50;
                font-size: 1.8em;
                margin-bottom: 20px;
                border-bottom: 3px solid #8b5cf6;
                padding-bottom: 10px;
            }}
            
            .trace-timeline {{
                position: relative;
                margin: 20px 0;
            }}
            
            .timeline-line {{
                position: absolute;
                left: 30px;
                top: 0;
                bottom: 0;
                width: 3px;
                background: linear-gradient(180deg, #8b5cf6 0%, #06b6d4 100%);
            }}
            
            .trace-item {{
                position: relative;
                margin: 0 0 20px 70px;
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #dee2e6;
            }}
            
            .trace-icon {{
                position: absolute;
                left: -55px;
                top: 20px;
                width: 30px;
                height: 30px;
                background: #8b5cf6;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                border: 3px solid white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            
            .trace-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            
            .trace-name {{
                font-size: 1.3em;
                font-weight: 600;
                color: #2c3e50;
            }}
            
            .trace-timestamp {{
                color: #666;
                font-size: 0.9em;
            }}
            
            .trace-details {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}
            
            .detail-item {{
                background: white;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
            }}
            
            .detail-label {{
                color: #666;
                font-size: 0.85em;
                text-transform: uppercase;
                margin-bottom: 5px;
            }}
            
            .detail-value {{
                color: #2c3e50;
                font-weight: 500;
            }}
            
            .observations-section {{
                margin-top: 20px;
                padding: 20px;
                background: #e3f2fd;
                border-radius: 8px;
            }}
            
            .observation-item {{
                background: white;
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #06b6d4;
            }}
            
            .observation-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }}
            
            .observation-name {{
                font-weight: 600;
                color: #0277bd;
            }}
            
            .observation-type {{
                background: #06b6d4;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                text-transform: uppercase;
            }}
            
            .usage-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                gap: 10px;
                margin-top: 10px;
            }}
            
            .usage-metric {{
                text-align: center;
                padding: 8px;
                background: #f5f5f5;
                border-radius: 4px;
            }}
            
            .usage-value {{
                font-weight: bold;
                color: #06b6d4;
            }}
            
            .usage-label {{
                font-size: 0.8em;
                color: #666;
            }}
            
            .no-data {{
                text-align: center;
                padding: 50px;
                color: #666;
                font-size: 1.1em;
            }}
            
            .metadata {{
                background: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.85em;
                margin-top: 10px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Langfuse Traces Dashboard</h1>
                <p>Complete Pipeline Run Observability - Session-Based Filtering</p>
                <p><strong>ZenML Pipeline:</strong> {session_metadata.get('pipeline_name', 'Unknown')} | <strong>Run:</strong> {session_metadata.get('run_name', 'Unknown')}</p>
                <p><strong>Pipeline Trace ID:</strong> {traces_data['pipeline_trace_id']}</p>
                <p><strong>Session ID:</strong> {traces_data['session_id'][:16]}... | <strong>Method:</strong> {retrieval_method.title()}</p>
                {f'<p><a href="{session_metadata.get("langfuse_session_url", "#")}" target="_blank" style="color: #ffffff; text-decoration: underline;">🔗 View in Langfuse</a></p>' if session_metadata.get("langfuse_session_url") else ""}
                <p><strong>Retrieved:</strong> {datetime.fromisoformat(traces_data['retrieval_timestamp']).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-value">{summary['total_traces']}</div>
                    <div class="summary-label">Total Traces</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{summary['total_observations']}</div>
                    <div class="summary-label">Observations</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{total_tokens:,}</div>
                    <div class="summary-label">Total Tokens</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">${total_cost:.4f}</div>
                    <div class="summary-label">Estimated Cost</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{len(summary['trace_types'])}</div>
                    <div class="summary-label">Trace Types</div>
                </div>
                <div class="summary-card">
                    <div class="summary-value">{traces_data['time_window']['minutes']}min</div>
                    <div class="summary-label">Time Window</div>
                </div>
            </div>
    """
    
    if traces:
        html_content += f"""
            <div class="content-section">
                <h2 class="section-title">🔍 Trace Timeline</h2>
                <div class="trace-timeline">
                    <div class="timeline-line"></div>
        """
        
        for i, trace in enumerate(traces):
            trace_observations = [obs for obs in observations if obs["trace_id"] == trace["id"]]
            
            html_content += f"""
                    <div class="trace-item">
                        <div class="trace-icon">{i+1}</div>
                        <div class="trace-header">
                            <div class="trace-name">{trace.get('name', 'Unnamed Trace')}</div>
                            <div class="trace-timestamp">
                                {datetime.fromisoformat(trace['timestamp']).strftime('%H:%M:%S') if trace.get('timestamp') else 'No timestamp'}
                            </div>
                        </div>
                        
                        <div class="trace-details">
                            <div class="detail-item">
                                <div class="detail-label">Trace ID</div>
                                <div class="detail-value">{trace['id'][:16]}...</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Session ID</div>
                                <div class="detail-value">{trace.get('session_id', 'None') or 'None'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">User ID</div>
                                <div class="detail-value">{trace.get('user_id', 'None') or 'None'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">Level</div>
                                <div class="detail-value">{trace.get('level', 'Unknown')}</div>
                            </div>
                        </div>
            """
            
            if trace.get('tags'):
                html_content += f"""
                        <div class="metadata">
                            <strong>Tags:</strong> {', '.join(trace['tags'])}
                        </div>
                """
            
            if trace_observations:
                html_content += f"""
                        <div class="observations-section">
                            <h4>🔬 Observations ({len(trace_observations)})</h4>
                """
                
                for obs in trace_observations[:10]:  # Limit to first 10 observations
                    usage_info = ""
                    if obs.get("usage") and isinstance(obs["usage"], dict):
                        usage = obs["usage"]
                        usage_info = f"""
                            <div class="usage-info">
                                <div class="usage-metric">
                                    <div class="usage-value">{usage.get('prompt_tokens', 0)}</div>
                                    <div class="usage-label">Input</div>
                                </div>
                                <div class="usage-metric">
                                    <div class="usage-value">{usage.get('completion_tokens', 0)}</div>
                                    <div class="usage-label">Output</div>
                                </div>
                                <div class="usage-metric">
                                    <div class="usage-value">{usage.get('total', 0)}</div>
                                    <div class="usage-label">Total</div>
                                </div>
                                <div class="usage-metric">
                                    <div class="usage-value">${usage.get('total_cost', 0.0):.4f}</div>
                                    <div class="usage-label">Cost</div>
                                </div>
                            </div>
                        """
                    
                    html_content += f"""
                            <div class="observation-item">
                                <div class="observation-header">
                                    <div class="observation-name">{obs.get('name', 'Unnamed Observation')}</div>
                                    <div class="observation-type">{obs.get('type', 'Unknown')}</div>
                                </div>
                                <div class="detail-value">
                                    <strong>Model:</strong> {obs.get('model', 'Not specified')}<br>
                                    <strong>Duration:</strong> {_calculate_duration(obs.get('start_time'), obs.get('end_time'))}
                                </div>
                                {usage_info}
                            </div>
                    """
                
                if len(trace_observations) > 10:
                    html_content += f"<p><em>... and {len(trace_observations) - 10} more observations</em></p>"
                
                html_content += "</div>"
            
            html_content += "</div>"
        
        html_content += """
                </div>
            </div>
        """
    else:
        html_content += """
            <div class="content-section">
                <div class="no-data">
                    <h3>No traces found in the specified time window</h3>
                    <p>Try increasing the time window or check your Langfuse configuration.</p>
                </div>
            </div>
        """
    
    html_content += """
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


def _calculate_duration(start_time: Optional[str], end_time: Optional[str]) -> str:
    """Calculate duration between start and end times."""
    if not start_time or not end_time:
        return "Unknown"
    
    try:
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        duration = end - start
        
        total_seconds = duration.total_seconds()
        if total_seconds < 1:
            return f"{total_seconds*1000:.0f}ms"
        else:
            return f"{total_seconds:.2f}s"
    except Exception:
        return "Unknown"