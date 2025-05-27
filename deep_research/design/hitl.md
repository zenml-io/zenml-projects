# Human Approval Step - Technical Specification

## Overview

Add a human-in-the-loop approval mechanism to the Deep Research Pipeline that allows users to review and approve/reject additional research recommendations before they are executed.

## Purpose

1. **Cost Control**: Prevent runaway token usage and API calls
2. **Quality Control**: Allow subject matter experts to guide research direction
3. **Transparency**: Show stakeholders what additional research is being considered
4. **Flexibility**: Enable selective approval of specific research queries

## Pipeline Integration Point

Since ZenML requires a static DAG, we need to:
1. Split the reflection step into two separate steps
2. Insert an approval step between them
3. Pass the approval decision to the second reflection step

### Current Pipeline Flow:
```python
# In parallel_research_pipeline.py
analyzed_state = cross_viewpoint_analysis_step(state=merged_state)
reflected_state = iterative_reflection_step(state=analyzed_state)
```

### New Pipeline Flow with Approval:
```python
# In parallel_research_pipeline.py
analyzed_state = cross_viewpoint_analysis_step(state=merged_state)

# Step 1: Generate reflection and recommendations (no searches yet)
reflection_output = generate_reflection_step(state=analyzed_state)

# Step 2: Get approval for recommended searches
approval_decision = get_research_approval_step(
    state=reflection_output.state,
    proposed_queries=reflection_output.recommended_queries,
    critique_points=reflection_output.critique_summary
)

# Step 3: Execute approved searches (if any)
reflected_state = execute_approved_searches_step(
    state=reflection_output.state,
    approval_decision=approval_decision,
    original_reflection=reflection_output
)
```

## Implementation Components

### 1. New Data Models

Add to `utils/pydantic_models.py`:
```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class ReflectionOutput(BaseModel):
    """Output from the reflection generation step."""
    state: ResearchState
    recommended_queries: List[str] = Field(default_factory=list)
    critique_summary: List[Dict[str, Any]] = Field(default_factory=list)
    additional_questions: List[str] = Field(default_factory=list)
    
class ApprovalDecision(BaseModel):
    """Approval decision from human reviewer."""
    approved: bool = False
    selected_queries: List[str] = Field(default_factory=list)
    approval_method: str = ""  # "APPROVE_ALL", "SKIP", "SELECT_SPECIFIC"
    reviewer_notes: str = ""
    timestamp: float = Field(default_factory=lambda: time.time())
```

### 2. Split Reflection into Two Steps

Create `steps/generate_reflection_step.py`:
```python
from typing import Annotated
from zenml import step
from utils.pydantic_models import ResearchState, ReflectionOutput

@step
def generate_reflection_step(
    state: ResearchState,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    reflection_prompt: str = REFLECTION_PROMPT,
) -> Annotated[ReflectionOutput, "reflection_output"]:
    """
    Generate reflection and recommendations WITHOUT executing searches.
    
    This step only analyzes the current state and produces recommendations.
    """
    logger.info("Generating reflection on research")
    
    # Existing reflection logic (from iterative_reflection_step)
    reflection_input = prepare_reflection_input(state)
    
    reflection_result = get_structured_llm_output(
        prompt=json.dumps(reflection_input),
        system_prompt=reflection_prompt,
        model=llm_model,
        fallback_response={"critique": [], "additional_questions": [], "recommended_search_queries": []}
    )
    
    # Return structured output for next steps
    return ReflectionOutput(
        state=state,
        recommended_queries=reflection_result.get("recommended_search_queries", []),
        critique_summary=reflection_result.get("critique", []),
        additional_questions=reflection_result.get("additional_questions", [])
    )
```

### 3. Approval Step

Create `steps/approval_step.py`:
```python
from typing import Annotated
from zenml import step
from zenml.alerter import Client
from utils.pydantic_models import ResearchState, ReflectionOutput, ApprovalDecision
import json

@step(enable_cache=False)  # Never cache approval decisions
def get_research_approval_step(
    reflection_output: ReflectionOutput,
    require_approval: bool = True,
    alerter_type: str = "slack",
    timeout: int = 3600
) -> Annotated[ApprovalDecision, "approval_decision"]:
    """
    Get human approval for additional research queries.
    
    Always returns an ApprovalDecision object. If require_approval is False,
    automatically approves all queries.
    """
    
    # If approval not required, auto-approve all
    if not require_approval:
        return ApprovalDecision(
            approved=True,
            selected_queries=reflection_output.recommended_queries,
            approval_method="AUTO_APPROVED",
            reviewer_notes="Approval not required by configuration"
        )
    
    # If no queries to approve, skip
    if not reflection_output.recommended_queries:
        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="NO_QUERIES",
            reviewer_notes="No additional queries recommended"
        )
    
    # Prepare approval request
    message = format_approval_request(
        main_query=reflection_output.state.main_query,
        progress_summary=summarize_research_progress(reflection_output.state),
        critique_points=reflection_output.critique_summary,
        proposed_queries=reflection_output.recommended_queries
    )
    
    try:
        # Get alerter and send request
        client = Client()
        response = client.active_stack.alerter.ask(
            message=message,
            params={"timeout": timeout}
        )
        
        # Parse response
        return parse_approval_response(response, reflection_output.recommended_queries)
        
    except Exception as e:
        logger.error(f"Approval request failed: {e}")
        # On error, default to not approved
        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="ERROR",
            reviewer_notes=f"Approval failed: {str(e)}"
        )
```

### 4. Execute Approved Searches Step

Create `steps/execute_approved_searches_step.py`:
```python
def format_approval_request(
    main_query: str,
    progress_summary: Dict[str, Any],
    critique_points: List[Dict[str, Any]],
    proposed_queries: List[str]
) -> str:
    """Format the approval request message."""
    
    # High-priority critiques
    high_priority = [c for c in critique_points if c.get("importance") == "high"]
    
    message = f"""
📊 **Research Progress Update**

**Main Query:** {main_query}

**Current Status:**
- Sub-questions analyzed: {progress_summary['completed_count']}
- Average confidence: {progress_summary['avg_confidence']}
- Low confidence areas: {progress_summary['low_confidence_count']}

**Key Issues Identified:**
{format_critique_summary(high_priority)}

**Proposed Additional Research** ({len(proposed_queries)} queries):
{format_query_list(proposed_queries)}

**Estimated Additional Time:** ~{len(proposed_queries) * 2} minutes
**Estimated Additional Cost:** ~${calculate_estimated_cost(proposed_queries)}

**Response Options:**
- Reply `APPROVE ALL` to proceed with all queries
- Reply `SKIP` to finish with current findings  
- Reply `SELECT 1,3,5` to approve specific queries by number

**Timeout:** Response required within {timeout//60} minutes
"""
    return message


```python
from typing import Annotated
from zenml import step
from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.pydantic_models import (
    ResearchState, ReflectionOutput, ApprovalDecision, 
    ReflectionMetadata, SynthesizedInfo
)

@step(output_materializers=ResearchStateMaterializer)
def execute_approved_searches_step(
    reflection_output: ReflectionOutput,
    approval_decision: ApprovalDecision,
    num_results_per_search: int = 3,
    cap_search_length: int = 20000,
    llm_model: str = "sambanova/DeepSeek-R1-Distill-Llama-70B",
    additional_synthesis_prompt: str = ADDITIONAL_SYNTHESIS_PROMPT,
) -> Annotated[ResearchState, "updated_state"]:
    """
    Execute approved searches and enhance the research state.
    
    This step receives the approval decision and only executes
    searches that were approved.
    """
    logger.info(f"Processing approval decision: {approval_decision.approval_method}")
    
    state = reflection_output.state
    enhanced_info = create_enhanced_info_copy(state.synthesized_info)
    
    # Check if we should execute searches
    if not approval_decision.approved or not approval_decision.selected_queries:
        logger.info("No additional searches approved")
        
        # Create metadata indicating no additional research
        reflection_metadata = ReflectionMetadata(
            critique_summary=[c.get("issue", "") for c in reflection_output.critique_summary],
            additional_questions_identified=reflection_output.additional_questions,
            searches_performed=[],
            improvements_made=0,
            user_decision=approval_decision.approval_method,
            reviewer_notes=approval_decision.reviewer_notes
        )
        
        state.update_after_reflection(enhanced_info, reflection_metadata)
        return state
    
    # Execute approved searches
    logger.info(f"Executing {len(approval_decision.selected_queries)} approved searches")
    
    for query in approval_decision.selected_queries:
        logger.info(f"Performing approved search: {query}")
        
        # Execute search (existing logic from iterative_reflection_step)
        search_results = search_and_extract_results(
            query=query,
            max_results=num_results_per_search,
            cap_content_length=cap_search_length,
        )
        
        # Find relevant sub-question and enhance
        # ... (rest of enhancement logic from original iterative_reflection_step)
    
    # Create final metadata with approval info
    reflection_metadata = ReflectionMetadata(
        critique_summary=[c.get("issue", "") for c in reflection_output.critique_summary],
        additional_questions_identified=reflection_output.additional_questions,
        searches_performed=approval_decision.selected_queries,
        improvements_made=count_improvements(enhanced_info),
        user_decision=approval_decision.approval_method,
        reviewer_notes=approval_decision.reviewer_notes
    )
    
    state.update_after_reflection(enhanced_info, reflection_metadata)
    return state
```

### 5. Updated Pipeline Definition

Update `pipelines/parallel_research_pipeline.py`:
```python
@step(output_materializers=ResearchStateMaterializer)
def iterative_reflection_step(
    state: ResearchState,
    max_additional_searches: int = 2,
    require_approval: bool = False,  # NEW
    approval_timeout: int = 3600,     # NEW
    alerter_type: str = "slack",      # NEW
    # ... other params
) -> Annotated[ResearchState, "updated_state"]:
    """Perform iterative reflection with optional human approval."""
    
    # ... existing reflection logic ...
    
    # Get recommended queries
    search_queries = reflection_result.get("recommended_search_queries", [])
    
    # NEW: Approval gate
    if require_approval and search_queries:
        approved, selected_queries = get_research_approval_step(
            state=state,
            proposed_queries=search_queries[:max_additional_searches],
            reflection_critique=reflection_result.get("critique", []),
            alerter_type=alerter_type,
            timeout=approval_timeout
        )
        
        if not approved:
            logger.info("Additional research not approved by user")
            # Create metadata indicating skipped research
            reflection_metadata = ReflectionMetadata(
                critique_summary=[item.get("issue", "") for item in reflection_result.get("critique", [])],
                additional_questions_identified=reflection_result.get("additional_questions", []),
                searches_performed=[],
                improvements_made=0,
                user_decision="SKIPPED_ADDITIONAL_RESEARCH"
            )
            state.update_after_reflection(state.synthesized_info, reflection_metadata)
            return state
        
        # Use only approved queries
        search_queries = selected_queries
    
    # ... continue with approved searches ...
```

## Testing Strategy

### Unit Tests for Approval Logic (`tests/test_approval_utils.py`):

Focus on testing the core approval parsing logic without running actual ZenML steps:

```python
import pytest
from utils.approval_utils import parse_approval_response
from utils.pydantic_models import ApprovalDecision

def test_parse_approval_responses():
    """Test parsing different approval responses."""
    queries = ["query1", "query2", "query3"]
    
    # Test approve all
    decision = parse_approval_response("APPROVE ALL", queries)
    assert decision.approved == True
    assert decision.selected_queries == queries
    assert decision.approval_method == "APPROVE_ALL"
    
    # Test skip
    decision = parse_approval_response("skip", queries)  # Test case insensitive
    assert decision.approved == False
    assert decision.selected_queries == []
    assert decision.approval_method == "SKIP"
    
    # Test selection
    decision = parse_approval_response("SELECT 1,3", queries)
    assert decision.approved == True
    assert decision.selected_queries == ["query1", "query3"]
    assert decision.approval_method == "SELECT_SPECIFIC"
    
    # Test invalid selection
    decision = parse_approval_response("SELECT invalid", queries)
    assert decision.approved == False
    assert decision.approval_method == "PARSE_ERROR"
    
    # Test out of range indices
    decision = parse_approval_response("SELECT 1,5,10", queries)
    assert decision.approved == True
    assert decision.selected_queries == ["query1"]  # Only valid indices
    assert decision.approval_method == "SELECT_SPECIFIC"


def test_format_approval_request():
    """Test formatting of approval request messages."""
    from utils.approval_utils import format_approval_request
    
    message = format_approval_request(
        main_query="Test query",
        progress_summary={
            'completed_count': 5,
            'avg_confidence': 0.75,
            'low_confidence_count': 2
        },
        critique_points=[
            {"issue": "Missing data", "importance": "high"},
            {"issue": "Minor gap", "importance": "low"}
        ],
        proposed_queries=["query1", "query2"]
    )
    
    assert "Test query" in message
    assert "5" in message
    assert "0.75" in message
    assert "2 queries" in message
    assert "APPROVE ALL" in message
    assert "SKIP" in message
    assert "SELECT" in message
```

### Note on Testing Approach:
- We focus on unit tests for the approval parsing and formatting logic
- Integration testing with actual ZenML steps and alerters will be done manually during development
- No full end-to-end integration tests are included to keep the test suite lightweight

## Key Changes Summary

### What Changed:
1. **Split `iterative_reflection_step` into 3 steps** to comply with ZenML's static DAG requirement
2. **Always execute all steps** - the approval step runs every time but auto-approves when `require_approval=False`
3. **Pass data between steps** using new Pydantic models (`ReflectionOutput`, `ApprovalDecision`)
4. **No conditionals in pipeline definition** - all branching logic moved inside steps

### New Files to Create:
- `steps/generate_reflection_step.py`
- `steps/approval_step.py`
- `steps/execute_approved_searches_step.py`
- `utils/approval_utils.py`
- Updates to `utils/pydantic_models.py`
- Updates to `pipelines/parallel_research_pipeline.py`

### Configuration:
- Add `require_approval` and `approval_timeout` to pipeline parameters
- Configure alerter in stack or via environment variables

### Usage:
```bash
# With approval enabled
python run.py --config configs/enhanced_research.yaml

# Without approval (default behavior)
python run.py --config configs/enhanced_research.yaml --no-approval
```

## Implementation Checklist

- [ ] Add new Pydantic models (`ReflectionOutput`, `ApprovalDecision`)
- [ ] Split existing `iterative_reflection_step` into `generate_reflection_step`
- [ ] Create `get_research_approval_step` 
- [ ] Create `execute_approved_searches_step`
- [ ] Update pipeline definition with new steps
- [ ] Add approval utility functions
- [ ] Configure Slack/Email alerter
- [ ] Add unit tests
- [ ] Add integration test with mocked alerter
- [ ] Update documentation
- [ ] Test end-to-end flow
