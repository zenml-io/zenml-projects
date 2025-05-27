# Prompt-Based Cost Visualization Design

## Overview

This document outlines the design for disaggregating LLM costs by prompt type in the ZenML Deep Research pipeline. The goal is to provide detailed insights into which prompts consume the most tokens and incur the highest costs, enabling better optimization decisions.

## Current State

### Tracing Structure
- **Traces**: One per pipeline run, containing all LLM calls
- **Observations**: Individual LLM calls with model, tokens, and cost data
- **Current Visualization**: Aggregates costs by model and step name only

### Problem
The current visualization shows total costs but doesn't break down spending by the specific prompt templates being used. This makes it difficult to:
1. Identify which prompts are most expensive
2. Optimize token usage for specific prompt types
3. Understand the cost distribution across different pipeline phases

## Proposed Solution

### 1. Prompt Type Identification

Create a mapping of prompt types to unique keywords that appear in each prompt template:

```python
PROMPT_IDENTIFIERS = {
    "query_decomposition": ["MAIN RESEARCH QUERY", "DIFFERENT DIMENSIONS", "sub-questions"],
    "search_query": ["Deep Research assistant", "effective search query"],
    "synthesis": ["information synthesis", "comprehensive answer", "confidence level"],
    "viewpoint_analysis": ["multi-perspective analysis", "viewpoint categories"],
    "reflection": ["critique and improve", "information gaps"],
    "additional_synthesis": ["enhance the original synthesis"],
    "conclusion_generation": ["Synthesis and Integration", "Direct Response to Main Query"],
    "executive_summary": ["executive summaries", "Key Findings", "250-400 words"],
    "introduction": ["engaging introductions", "Context and Relevance"],
}
```

Note: From the sample observation, the system prompt is accessed via:
- `observation.input['messages'][0]['content']` for the system prompt
- `observation.input['messages'][1]['content']` for the user input
- Token usage is in `observation.usage.input` and `observation.usage.output`
- Cost is in `observation.calculated_total_cost` (defaults to 0.0)

### 2. New Utility Functions

Add to `utils/tracing_metadata_utils.py`:

```python
def identify_prompt_type(observation: ObservationsView) -> Optional[str]:
    """
    Identify the prompt type based on keywords in the observation's input.
    
    Examines the system prompt in observation.input['messages'][0]['content']
    for unique keywords that identify each prompt type.
    
    Returns:
        str: The prompt type name, or "unknown" if not identified
    """
    
def get_costs_by_prompt_type(trace_id: str) -> Dict[str, Dict[str, float]]:
    """
    Get cost breakdown by prompt type for a given trace.
    
    Uses observation.usage.input/output for token counts and
    observation.calculated_total_cost for costs.
    
    Returns:
        Dict mapping prompt_type to {
            'cost': float,
            'input_tokens': int,
            'output_tokens': int,
            'count': int  # number of calls
        }
    """
    
def get_prompt_type_statistics(trace_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed statistics for each prompt type.
    
    Returns:
        Dict mapping prompt_type to {
            'cost': float,
            'input_tokens': int,
            'output_tokens': int,
            'count': int,
            'avg_cost_per_call': float,
            'avg_input_tokens': float,
            'avg_output_tokens': float,
            'percentage_of_total_cost': float
        }
    """
```

### 3. Visualization Updates

#### A. Data Collection
Update `steps/collect_tracing_metadata_step.py` to:

1. Call `get_costs_by_prompt_type()` for the trace
2. Calculate percentages and averages
3. Store prompt-level metrics in the `TracingMetadata` model

#### B. Add to Pydantic Model
Update `utils/pydantic_models.py`:

```python
class PromptTypeMetrics(BaseModel):
    """Metrics for a specific prompt type."""
    prompt_type: str
    total_cost: float
    input_tokens: int
    output_tokens: int
    call_count: int
    avg_cost_per_call: float
    percentage_of_total_cost: float

class TracingMetadata(BaseModel):
    # ... existing fields ...
    prompt_metrics: List[PromptTypeMetrics] = Field(
        default_factory=list,
        description="Cost breakdown by prompt type"
    )
```

#### C. HTML Visualization
Update `materializers/tracing_metadata_materializer.py` to add:

1. **Bar Chart**: Cost by prompt type
   - X-axis: Prompt types
   - Y-axis: Cost in USD
   - Color-coded bars with hover tooltips

2. **Token Usage Chart**: Stacked bar chart
   - X-axis: Prompt types
   - Y-axis: Token count
   - Stacked: Input tokens (bottom) and output tokens (top)

3. **Efficiency Table**: 
   - Columns: Prompt Type, Total Cost, Calls, Avg Cost/Call, % of Total
   - Sortable by any column
   - Highlight most expensive prompts

### 4. Implementation Approach

#### Phase 1: Core Functionality
1. Implement `identify_prompt_type()` with robust keyword matching
   - Access system prompt via `observation.input['messages'][0]['content']`
   - Handle cases where messages structure differs
   - Add fallback logic for observations without clear prompt type
2. Test with sample traces to ensure accurate categorization

#### Phase 2: Cost Aggregation
1. Implement `get_costs_by_prompt_type()` 
   - Use `observation.usage.input` and `observation.usage.output` for token counts
   - Use `observation.calculated_total_cost` for cost (fallback to 0.0 if None)
   - Handle edge cases (missing cost data, partial tokens)
2. Add caching for performance with large traces

#### Phase 3: Visualization
1. Update data models
2. Implement chart generation using Chart.js or similar
3. Add interactive features (sorting, filtering)

### 5. Visualization Mockup

```
┌─────────────────────────────────────────────────────────┐
│                 Cost by Prompt Type                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  $0.50 ┤ ████                                          │
│  $0.40 ┤ ████  ████                                    │
│  $0.30 ┤ ████  ████  ████                              │
│  $0.20 ┤ ████  ████  ████  ████                        │
│  $0.10 ┤ ████  ████  ████  ████  ████                  │
│  $0.00 └──────────────────────────────────────         │
│         Query  Synth  Search Reflect  Exec             │
│         Decomp                        Summary           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Prompt Type Efficiency                      │
├────────────────┬──────┬──────┬───────────┬────────────┤
│ Prompt Type    │ Cost │ Calls│ Avg $/Call│ % of Total │
├────────────────┼──────┼──────┼───────────┼────────────┤
│ Query Decomp   │$0.45 │   3  │   $0.15   │    28%     │
│ Synthesis      │$0.38 │  12  │   $0.03   │    24%     │
│ Search Query   │$0.25 │  45  │   $0.006  │    16%     │
│ Reflection     │$0.20 │   8  │   $0.025  │    13%     │
│ Executive Sum  │$0.15 │   1  │   $0.15   │     9%     │
└────────────────┴──────┴──────┴───────────┴────────────┘
```

### 6. Future Enhancements

1. **Drill-down capability**: Click on a prompt type to see individual observations
2. **Time-series analysis**: Track prompt costs over multiple pipeline runs
3. **Optimization suggestions**: Automatically identify prompts that could be shortened
4. **A/B testing support**: Compare costs between different prompt versions
5. **Export functionality**: Download cost data as CSV/JSON

### 7. Configuration

Add to pipeline configs:

```yaml
cost_visualization:
  group_by_prompt: true
  show_token_breakdown: true
  highlight_threshold: 0.1  # Highlight prompts > 10% of total cost
```

## Benefits

1. **Cost Transparency**: Clear understanding of where money is being spent
2. **Optimization Targets**: Identify which prompts to optimize first
3. **Token Efficiency**: See which prompts generate the most output relative to input
4. **Budget Planning**: Better estimates for future research tasks
5. **Prompt Engineering**: Data-driven approach to prompt refinement

## Testing Strategy

1. Unit tests for prompt identification logic
2. Integration tests with sample Langfuse data
3. Visualization tests using snapshot testing
4. Performance tests with large traces (1000+ observations)

## Migration Plan

Since this is an additive feature:
1. No breaking changes to existing code
2. Gradual rollout: Start with basic prompt identification
3. Feature flag for new visualizations
4. Backward compatibility with traces lacking prompt data