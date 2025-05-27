#!/usr/bin/env python3
"""Test script for prompt cost visualization feature."""

import sys
sys.path.append('..')

from utils.tracing_metadata_utils import (
    identify_prompt_type,
    get_costs_by_prompt_type,
    get_prompt_type_statistics,
    PROMPT_IDENTIFIERS
)
from utils.pydantic_models import PromptTypeMetrics, TracingMetadata
from langfuse.client import ObservationsView


# Mock observation data for testing
class MockUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input = input_tokens
        self.output = output_tokens


class MockObservation:
    def __init__(self, prompt_type_content, input_tokens=100, output_tokens=50, cost=0.01):
        self.input = {
            'messages': [
                {'content': prompt_type_content},
                {'content': 'user message'}
            ]
        }
        self.usage = MockUsage(input_tokens, output_tokens)
        self.calculated_total_cost = cost


def test_identify_prompt_type():
    """Test the identify_prompt_type function."""
    print("Testing identify_prompt_type...")
    
    # Test each prompt type
    test_cases = [
        ("You are a Deep Research assistant specializing in effective search query generation.", "search_query"),
        ("Given the MAIN RESEARCH QUERY and DIFFERENT DIMENSIONS sub-questions", "query_decomposition"),
        ("Your task is information synthesis with comprehensive answer and confidence level", "synthesis"),
        ("This is an unknown prompt type", "unknown"),
    ]
    
    for content, expected in test_cases:
        obs = MockObservation(content)
        result = identify_prompt_type(obs)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Content: '{content[:50]}...' => {result} (expected: {expected})")


def test_prompt_metrics_creation():
    """Test creating PromptTypeMetrics objects."""
    print("\nTesting PromptTypeMetrics creation...")
    
    metrics = PromptTypeMetrics(
        prompt_type="search_query",
        total_cost=0.25,
        input_tokens=5000,
        output_tokens=2000,
        call_count=10,
        avg_cost_per_call=0.025,
        percentage_of_total_cost=35.5
    )
    
    print(f"  ✓ Created metrics for {metrics.prompt_type}")
    print(f"    Total cost: ${metrics.total_cost:.4f}")
    print(f"    Calls: {metrics.call_count}")
    print(f"    Avg cost/call: ${metrics.avg_cost_per_call:.4f}")
    print(f"    % of total: {metrics.percentage_of_total_cost:.1f}%")


def test_tracing_metadata_with_prompts():
    """Test TracingMetadata with prompt metrics."""
    print("\nTesting TracingMetadata with prompt metrics...")
    
    # Create sample prompt metrics
    prompt_metrics = [
        PromptTypeMetrics(
            prompt_type="query_decomposition",
            total_cost=0.45,
            input_tokens=3000,
            output_tokens=1500,
            call_count=3,
            avg_cost_per_call=0.15,
            percentage_of_total_cost=28.0
        ),
        PromptTypeMetrics(
            prompt_type="synthesis",
            total_cost=0.38,
            input_tokens=8000,
            output_tokens=4000,
            call_count=12,
            avg_cost_per_call=0.032,
            percentage_of_total_cost=24.0
        ),
        PromptTypeMetrics(
            prompt_type="search_query",
            total_cost=0.25,
            input_tokens=5000,
            output_tokens=2000,
            call_count=45,
            avg_cost_per_call=0.006,
            percentage_of_total_cost=16.0
        ),
    ]
    
    # Create TracingMetadata
    metadata = TracingMetadata(
        pipeline_run_name="test-pipeline-run",
        pipeline_run_id="test-id-123",
        total_cost=1.58,
        total_input_tokens=20000,
        total_output_tokens=10000,
        prompt_metrics=prompt_metrics
    )
    
    print(f"  ✓ Created TracingMetadata with {len(metadata.prompt_metrics)} prompt types")
    print(f"    Total pipeline cost: ${metadata.total_cost:.4f}")
    print("\n  Prompt breakdown:")
    for metric in metadata.prompt_metrics:
        print(f"    - {metric.prompt_type}: ${metric.total_cost:.4f} ({metric.percentage_of_total_cost:.1f}%)")


def test_visualization_html_generation():
    """Test that visualization HTML can be generated."""
    print("\nTesting HTML visualization generation...")
    
    from materializers.tracing_metadata_materializer import TracingMetadataMaterializer
    
    # Create metadata with prompt metrics
    metadata = TracingMetadata(
        pipeline_run_name="test-visualization",
        pipeline_run_id="test-viz-123",
        total_cost=2.50,
        total_input_tokens=30000,
        total_output_tokens=15000,
        total_tokens=45000,
        formatted_latency="2m 30.5s",
        models_used=["gpt-4", "claude-3"],
        prompt_metrics=[
            PromptTypeMetrics(
                prompt_type="synthesis",
                total_cost=1.20,
                input_tokens=15000,
                output_tokens=8000,
                call_count=10,
                avg_cost_per_call=0.12,
                percentage_of_total_cost=48.0
            ),
            PromptTypeMetrics(
                prompt_type="search_query",
                total_cost=0.80,
                input_tokens=10000,
                output_tokens=5000,
                call_count=50,
                avg_cost_per_call=0.016,
                percentage_of_total_cost=32.0
            ),
        ]
    )
    
    materializer = TracingMetadataMaterializer(uri="/tmp/test")
    html = materializer._generate_visualization_html(metadata)
    
    # Check that key elements are present
    checks = [
        ("Cost Analysis by Prompt Type" in html, "Prompt cost section"),
        ("promptCostChart" in html, "Cost chart canvas"),
        ("promptTokenChart" in html, "Token chart canvas"),
        ("Chart.js" in html, "Chart.js library"),
        ("Prompt Type Efficiency" in html, "Efficiency table"),
        ("Synthesis" in html, "Synthesis prompt type"),
        ("Search Query" in html, "Search Query prompt type"),
    ]
    
    for check, description in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {description}")
    
    # Save HTML for manual inspection if needed
    with open("/tmp/test_visualization.html", "w") as f:
        f.write(html)
    print("\n  ℹ️  HTML saved to /tmp/test_visualization.html for manual inspection")


def main():
    """Run all tests."""
    print("=== Prompt Cost Visualization Tests ===\n")
    
    test_identify_prompt_type()
    test_prompt_metrics_creation()
    test_tracing_metadata_with_prompts()
    test_visualization_html_generation()
    
    print("\n=== All tests completed! ===")


if __name__ == "__main__":
    main()