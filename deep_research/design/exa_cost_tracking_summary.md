# Exa Cost Tracking Implementation Summary

## Overview
This implementation adds comprehensive cost tracking for Exa search queries throughout the ZenML Deep Research pipeline. The costs are tracked at every step where searches are performed and aggregated for final visualization alongside LLM costs.

## Key Changes

### 1. Core Infrastructure
- **`utils/search_utils.py`**: Modified `exa_search()` to extract cost from `response.cost_dollars.total`
- **`utils/search_utils.py`**: Updated `extract_search_results()` and `search_and_extract_results()` to return tuple `(results, cost)`
- **`utils/pydantic_models.py`**: Added search cost tracking fields to `ResearchState`:
  - `search_costs: Dict[str, float]` - Total costs by provider
  - `search_cost_details: List[Dict[str, Any]]` - Detailed cost logs

### 2. Pipeline Steps
Updated all steps that perform searches to handle the new cost tracking:

- **`process_sub_question_step.py`**: Tracks costs for sub-question searches
- **`execute_approved_searches_step.py`**: Tracks costs for reflection-based searches
- **`iterative_reflection_step.py`**: Tracks costs for gap-filling searches
- **`merge_results_step.py`**: Aggregates costs from parallel sub-states

Each step:
1. Unpacks the tuple from `search_and_extract_results()`
2. Updates `state.search_costs["exa"]` with cumulative cost
3. Appends detailed cost information to `state.search_cost_details`

### 3. Metadata Collection
- **`utils/pydantic_models.py`**: Added search cost fields to `TracingMetadata`
- **`collect_tracing_metadata_step.py`**: Extracts search costs from final state and includes them in tracing metadata

### 4. Visualization
- **`materializers/tracing_metadata_materializer.py`**: Enhanced to display:
  - Individual search provider costs with query counts
  - Combined cost summary (LLM + Search)
  - Interactive doughnut chart showing cost breakdown
  - Percentage calculations for cost distribution

## Usage

When running a pipeline with Exa as the search provider:

```python
# In pipeline configuration
search_provider="exa"  # or "both" to use Exa alongside Tavily
```

The pipeline will automatically:
1. Track costs for each Exa search query
2. Aggregate costs across all steps
3. Display total costs in the final visualization

## Cost Information Captured

For each search, the system captures:
- Provider name (e.g., "exa")
- Search query text
- Cost in dollars
- Timestamp
- Pipeline step name
- Purpose (e.g., "sub_question", "reflection_enhancement", "gap_filling")
- Related sub-question (if applicable)

## Example Output

In the final HTML visualization:
```
Search Provider Costs
EXA Search: $0.0280
10 queries • $0.0028/query

Combined Cost Summary
LLM Cost: $0.1234 (81.5% of total)
Search Cost: $0.0280 (18.5% of total)
Total Pipeline Cost: $0.1514
```

## Testing

Run the test script to verify the implementation:
```bash
python design/test_exa_cost_tracking.py
```

All tests should pass, confirming:
- Exa API cost extraction works correctly
- ResearchState properly tracks costs
- Cost aggregation across steps functions properly

## Notes

- Tavily doesn't provide cost information in their API, so only Exa costs are tracked
- Costs are tracked even if searches fail (cost is still incurred)
- The implementation is backward compatible - pipelines without Exa will simply show no search costs