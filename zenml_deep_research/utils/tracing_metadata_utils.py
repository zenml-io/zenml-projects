import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

from langfuse import Langfuse
from langfuse.api.core import ApiError
from langfuse.client import ObservationsView, TraceWithDetails
from rich import print
from rich.console import Console
from rich.table import Table

console = Console()

langfuse = Langfuse()

# Rate limiting configuration
# Adjust these based on your Langfuse tier:
# - Hobby: 30 req/min for Other APIs -> ~2s between requests
# - Core: 100 req/min -> ~0.6s between requests
# - Pro: 1000 req/min -> ~0.06s between requests
RATE_LIMIT_DELAY = 0.1  # 100ms between requests (safe for most tiers)
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # Initial backoff in seconds

# Batch processing configuration
BATCH_DELAY = 0.5  # Additional delay between batches of requests


def rate_limited(func):
    """Decorator to add rate limiting between API calls."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        time.sleep(RATE_LIMIT_DELAY)
        return func(*args, **kwargs)

    return wrapper


def retry_with_backoff(func):
    """Decorator to retry functions with exponential backoff on rate limit errors."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        backoff = INITIAL_BACKOFF
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except ApiError as e:
                if e.status_code == 429:  # Rate limit error
                    last_exception = e
                    if attempt < MAX_RETRIES - 1:
                        wait_time = backoff * (2**attempt)
                        console.print(
                            f"[yellow]Rate limit hit. Retrying in {wait_time:.1f}s...[/yellow]"
                        )
                        time.sleep(wait_time)
                        continue
                raise
            except Exception:
                # For non-rate limit errors, raise immediately
                raise

        # If we've exhausted all retries
        if last_exception:
            raise last_exception

    return wrapper


@rate_limited
@retry_with_backoff
def fetch_traces_safe(limit: Optional[int] = None) -> List[TraceWithDetails]:
    """Safely fetch traces with rate limiting and retry logic."""
    return langfuse.fetch_traces(limit=limit).data


@rate_limited
@retry_with_backoff
def fetch_observations_safe(trace_id: str) -> List[ObservationsView]:
    """Safely fetch observations with rate limiting and retry logic."""
    return langfuse.fetch_observations(trace_id=trace_id).data


def get_total_trace_cost(trace_id: str) -> float:
    """Calculate the total cost for a single trace by summing all observation costs.

    Args:
        trace_id: The ID of the trace to calculate cost for

    Returns:
        Total cost across all observations in the trace
    """
    try:
        observations = fetch_observations_safe(trace_id=trace_id)
        total_cost = 0.0

        for obs in observations:
            # Check multiple possible cost fields
            if (
                hasattr(obs, "calculated_total_cost")
                and obs.calculated_total_cost
            ):
                total_cost += obs.calculated_total_cost
            elif hasattr(obs, "total_price") and obs.total_price:
                total_cost += obs.total_price
            elif hasattr(obs, "total_cost") and obs.total_cost:
                total_cost += obs.total_cost
            # If cost details are available, calculate from input/output costs
            elif hasattr(obs, "calculated_input_cost") and hasattr(
                obs, "calculated_output_cost"
            ):
                if obs.calculated_input_cost and obs.calculated_output_cost:
                    total_cost += (
                        obs.calculated_input_cost + obs.calculated_output_cost
                    )

        return total_cost
    except Exception as e:
        print(f"[red]Error calculating trace cost: {e}[/red]")
        return 0.0


def get_total_tokens_used(trace_id: str) -> Tuple[int, int]:
    """Calculate total input and output tokens used for a trace.

    Args:
        trace_id: The ID of the trace to calculate tokens for

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    try:
        observations = fetch_observations_safe(trace_id=trace_id)
        total_input_tokens = 0
        total_output_tokens = 0

        for obs in observations:
            # Check for token fields in different possible locations
            if hasattr(obs, "usage") and obs.usage:
                if hasattr(obs.usage, "input") and obs.usage.input:
                    total_input_tokens += obs.usage.input
                if hasattr(obs.usage, "output") and obs.usage.output:
                    total_output_tokens += obs.usage.output
            # Also check for direct token fields
            elif hasattr(obs, "promptTokens") and hasattr(
                obs, "completionTokens"
            ):
                if obs.promptTokens:
                    total_input_tokens += obs.promptTokens
                if obs.completionTokens:
                    total_output_tokens += obs.completionTokens

        return total_input_tokens, total_output_tokens
    except Exception as e:
        print(f"[red]Error calculating tokens: {e}[/red]")
        return 0, 0


def get_trace_stats(trace: TraceWithDetails) -> Dict[str, Any]:
    """Get comprehensive statistics for a trace.

    Args:
        trace: The trace object to analyze

    Returns:
        Dictionary containing trace statistics including cost, latency, tokens, and metadata
    """
    try:
        # Get cost and token data
        total_cost = get_total_trace_cost(trace.id)
        input_tokens, output_tokens = get_total_tokens_used(trace.id)

        # Get observation count
        observations = fetch_observations_safe(trace_id=trace.id)
        observation_count = len(observations)

        # Extract model information from observations
        models_used = set()
        for obs in observations:
            if hasattr(obs, "model") and obs.model:
                models_used.add(obs.model)

        stats = {
            "trace_id": trace.id,
            "timestamp": trace.timestamp,
            "total_cost": total_cost,
            "latency_seconds": trace.latency
            if hasattr(trace, "latency")
            else 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "observation_count": observation_count,
            "models_used": list(models_used),
            "metadata": trace.metadata if hasattr(trace, "metadata") else {},
            "tags": trace.tags if hasattr(trace, "tags") else [],
            "user_id": trace.user_id if hasattr(trace, "user_id") else None,
            "session_id": trace.session_id
            if hasattr(trace, "session_id")
            else None,
        }

        # Add formatted latency
        if stats["latency_seconds"]:
            minutes = int(stats["latency_seconds"] // 60)
            seconds = stats["latency_seconds"] % 60
            stats["latency_formatted"] = f"{minutes}m {seconds:.1f}s"
        else:
            stats["latency_formatted"] = "0m 0.0s"

        return stats
    except Exception as e:
        print(f"[red]Error getting trace stats: {e}[/red]")
        return {}


def get_traces_by_name(name: str, limit: int = 1) -> List[TraceWithDetails]:
    """Get traces by name using Langfuse API.

    Args:
        name: The name of the trace to search for
        limit: Maximum number of traces to return (default: 1)

    Returns:
        List of traces matching the name
    """
    try:
        # Use the Langfuse API to get traces by name
        traces_response = langfuse.get_traces(name=name, limit=limit)
        return traces_response.data
    except Exception as e:
        print(f"[red]Error fetching traces by name: {e}[/red]")
        return []


def get_observations_for_trace(trace_id: str) -> List[ObservationsView]:
    """Get all observations for a specific trace.

    Args:
        trace_id: The ID of the trace

    Returns:
        List of observations for the trace
    """
    try:
        observations_response = langfuse.get_observations(trace_id=trace_id)
        return observations_response.data
    except Exception as e:
        print(f"[red]Error fetching observations: {e}[/red]")
        return []


def filter_traces_by_date_range(
    start_date: datetime, end_date: datetime, limit: Optional[int] = None
) -> List[TraceWithDetails]:
    """Filter traces within a specific date range.

    Args:
        start_date: Start of the date range (inclusive)
        end_date: End of the date range (inclusive)
        limit: Maximum number of traces to return

    Returns:
        List of traces within the date range
    """
    try:
        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Fetch all traces (or up to API maximum limit of 100)
        all_traces = fetch_traces_safe(limit=limit or 100)

        # Filter by date range
        filtered_traces = [
            trace
            for trace in all_traces
            if start_date <= trace.timestamp <= end_date
        ]

        # Sort by timestamp (most recent first)
        filtered_traces.sort(key=lambda x: x.timestamp, reverse=True)

        # Apply limit if specified
        if limit:
            filtered_traces = filtered_traces[:limit]

        return filtered_traces
    except Exception as e:
        print(f"[red]Error filtering traces by date range: {e}[/red]")
        return []


def get_traces_last_n_days(
    days: int, limit: Optional[int] = None
) -> List[TraceWithDetails]:
    """Get traces from the last N days.

    Args:
        days: Number of days to look back
        limit: Maximum number of traces to return

    Returns:
        List of traces from the last N days
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    return filter_traces_by_date_range(start_date, end_date, limit)


def get_trace_stats_batch(
    traces: List[TraceWithDetails], show_progress: bool = True
) -> List[Dict[str, Any]]:
    """Get statistics for multiple traces efficiently with progress tracking.

    Args:
        traces: List of traces to analyze
        show_progress: Whether to show progress bar

    Returns:
        List of dictionaries containing trace statistics
    """
    stats_list = []

    for i, trace in enumerate(traces):
        if show_progress and i % 5 == 0:
            console.print(
                f"[dim]Processing trace {i + 1}/{len(traces)}...[/dim]"
            )

        stats = get_trace_stats(trace)
        stats_list.append(stats)

    return stats_list


def get_aggregate_stats_for_traces(
    traces: List[TraceWithDetails],
) -> Dict[str, Any]:
    """Calculate aggregate statistics for a list of traces.

    Args:
        traces: List of traces to analyze

    Returns:
        Dictionary containing aggregate statistics
    """
    if not traces:
        return {
            "trace_count": 0,
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "average_cost_per_trace": 0.0,
            "average_latency_seconds": 0.0,
            "total_observations": 0,
        }

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0
    total_observations = 0
    all_models = set()

    for trace in traces:
        stats = get_trace_stats(trace)
        total_cost += stats.get("total_cost", 0)
        total_input_tokens += stats.get("input_tokens", 0)
        total_output_tokens += stats.get("output_tokens", 0)
        total_latency += stats.get("latency_seconds", 0)
        total_observations += stats.get("observation_count", 0)
        all_models.update(stats.get("models_used", []))

    return {
        "trace_count": len(traces),
        "total_cost": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "average_cost_per_trace": total_cost / len(traces) if traces else 0,
        "average_latency_seconds": total_latency / len(traces)
        if traces
        else 0,
        "total_observations": total_observations,
        "models_used": list(all_models),
    }


def display_trace_stats_table(
    traces: List[TraceWithDetails], title: str = "Trace Statistics"
):
    """Display trace statistics in a formatted table.

    Args:
        traces: List of traces to display
        title: Title for the table
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    table.add_column("Timestamp", style="yellow")
    table.add_column("Cost ($)", justify="right", style="green")
    table.add_column("Tokens (In/Out)", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Observations", justify="right")

    for trace in traces[:10]:  # Limit to 10 for display
        stats = get_trace_stats(trace)
        table.add_row(
            stats["trace_id"][:12] + "...",
            stats["timestamp"].strftime("%Y-%m-%d %H:%M"),
            f"${stats['total_cost']:.4f}",
            f"{stats['input_tokens']:,}/{stats['output_tokens']:,}",
            stats["latency_formatted"],
            str(stats["observation_count"]),
        )

    console.print(table)


if __name__ == "__main__":
    print(
        "[bold cyan]ZenML Deep Research - Tracing Metadata Utilities Demo[/bold cyan]\n"
    )

    try:
        # Fetch recent traces
        print("[yellow]Fetching recent traces...[/yellow]")
        traces = fetch_traces_safe(limit=5)

        if not traces:
            print("[red]No traces found![/red]")
            exit(1)
    except ApiError as e:
        if e.status_code == 429:
            print("[red]Rate limit exceeded. Please try again later.[/red]")
            print(
                "[yellow]Tip: Consider upgrading your Langfuse tier for higher rate limits.[/yellow]"
            )
        else:
            print(f"[red]API Error: {e}[/red]")
        exit(1)
    except Exception as e:
        print(f"[red]Error fetching traces: {e}[/red]")
        exit(1)

    # Demo 1: Get stats for a single trace
    print("\n[bold]1. Single Trace Statistics:[/bold]")
    first_trace = traces[0]
    stats = get_trace_stats(first_trace)

    console.print(f"Trace ID: [cyan]{stats['trace_id']}[/cyan]")
    console.print(f"Timestamp: [yellow]{stats['timestamp']}[/yellow]")
    console.print(f"Total Cost: [green]${stats['total_cost']:.4f}[/green]")
    console.print(
        f"Tokens - Input: [blue]{stats['input_tokens']:,}[/blue], Output: [blue]{stats['output_tokens']:,}[/blue]"
    )
    console.print(f"Latency: [magenta]{stats['latency_formatted']}[/magenta]")
    console.print(f"Observations: [white]{stats['observation_count']}[/white]")
    console.print(
        f"Models Used: [cyan]{', '.join(stats['models_used'])}[/cyan]"
    )

    # Demo 2: Get traces from last 7 days
    print("\n[bold]2. Traces from Last 7 Days:[/bold]")
    recent_traces = get_traces_last_n_days(7, limit=10)
    print(
        f"Found [green]{len(recent_traces)}[/green] traces in the last 7 days"
    )

    if recent_traces:
        display_trace_stats_table(recent_traces, "Last 7 Days Traces")

    # Demo 3: Filter traces by date range
    print("\n[bold]3. Filter Traces by Date Range:[/bold]")
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3)

    filtered_traces = filter_traces_by_date_range(start_date, end_date)
    print(
        f"Found [green]{len(filtered_traces)}[/green] traces between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
    )

    # Demo 4: Aggregate statistics
    print("\n[bold]4. Aggregate Statistics for All Recent Traces:[/bold]")
    agg_stats = get_aggregate_stats_for_traces(traces)

    table = Table(
        title="Aggregate Statistics",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="yellow")

    table.add_row("Total Traces", str(agg_stats["trace_count"]))
    table.add_row("Total Cost", f"${agg_stats['total_cost']:.4f}")
    table.add_row(
        "Average Cost per Trace", f"${agg_stats['average_cost_per_trace']:.4f}"
    )
    table.add_row("Total Input Tokens", f"{agg_stats['total_input_tokens']:,}")
    table.add_row(
        "Total Output Tokens", f"{agg_stats['total_output_tokens']:,}"
    )
    table.add_row("Total Tokens", f"{agg_stats['total_tokens']:,}")
    table.add_row(
        "Average Latency", f"{agg_stats['average_latency_seconds']:.1f}s"
    )
    table.add_row("Total Observations", str(agg_stats["total_observations"]))

    console.print(table)

    # Demo 5: Cost breakdown by observation
    print("\n[bold]5. Cost Breakdown for First Trace:[/bold]")
    observations = fetch_observations_safe(trace_id=first_trace.id)

    if observations:
        table = Table(
            title="Observation Cost Breakdown",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Observation", style="cyan", no_wrap=True)
        table.add_column("Model", style="yellow")
        table.add_column("Tokens (In/Out)", justify="right")
        table.add_column("Cost", justify="right", style="green")

        for i, obs in enumerate(observations[:5]):  # Show first 5
            cost = 0.0
            if hasattr(obs, "calculated_total_cost"):
                cost = obs.calculated_total_cost or 0.0

            in_tokens = 0
            out_tokens = 0
            if hasattr(obs, "usage") and obs.usage:
                in_tokens = obs.usage.input or 0
                out_tokens = obs.usage.output or 0
            elif hasattr(obs, "promptTokens"):
                in_tokens = obs.promptTokens or 0
                out_tokens = obs.completionTokens or 0

            table.add_row(
                f"Obs {i + 1}",
                obs.model if hasattr(obs, "model") else "Unknown",
                f"{in_tokens:,}/{out_tokens:,}",
                f"${cost:.4f}",
            )

        console.print(table)
