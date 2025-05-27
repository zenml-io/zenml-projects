# Design Document: Migrating from Dataclasses to Pydantic Models in ZenML Deep Research

## Overview

This document outlines a plan to migrate the current dataclass-based state objects to Pydantic models in the ZenML Deep Research project. The migration will improve type validation, simplify serialization, leverage ZenML's built-in Pydantic support, and enable explicit `HTMLString` artifacts at key pipeline steps.

## Current State

The project currently uses:
- Dataclasses for state objects (`ResearchState`, `SearchResult`, etc.)
- Custom serialization/deserialization with `_convert_to_dict()` and `_convert_from_dict()`
- A custom `ResearchStateMaterializer` with 350+ lines of code

## Migration Goals

- [x] Replace dataclasses with Pydantic models for better validation and error messages
- [x] Leverage ZenML's built-in `PydanticMaterializer` to simplify serialization
- [x] Make HTML generation a first-class artifact where appropriate
- [x] Maintain existing visualizations while reducing code complexity

## Implementation Plan

### 1. Create Pydantic Model Equivalents
- [x] Convert leaf models first (`SearchResult`, `ViewpointTension`, etc.)
- [x] Implement nested models (`SynthesizedInfo`, `ViewpointAnalysis`, etc.)
- [x] Finally convert the main `ResearchState` model

For each model, we'll follow this pattern:

```python
# Before (dataclass)
@dataclass
class SearchResult:
    url: str = ""
    content: str = ""
    title: str = ""
    snippet: str = ""

# After (Pydantic)
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """Represents a search result for a sub-question."""
    url: str = ""
    content: str = ""
    title: str = ""
    snippet: str = ""
    
    model_config = {
        "extra": "ignore",      # Ignore extra fields during deserialization
        "frozen": False,        # Allow attribute updates
        "validate_assignment": True,  # Validate when attributes are set
    }
```

The main `ResearchState` model will require special attention to handle the update methods correctly:

```python
class ResearchState(BaseModel):
    # Base fields
    main_query: str = ""
    sub_questions: List[str] = Field(default_factory=list)
    search_results: Dict[str, List[SearchResult]] = Field(default_factory=dict)
    # ...other fields...
    
    model_config = {
        "validate_assignment": True,
        "frozen": False,
    }
    
    def get_current_stage(self) -> str:
        """Determine the current stage of research based on filled data."""
        if self.final_report_html:
            return "final_report"
        # ...rest of implementation...
        
    def update_sub_questions(self, sub_questions: List[str]) -> None:
        """Update the sub-questions list."""
        self.sub_questions = sub_questions
```

### 2. Create Extended PydanticMaterializer
- [x] Create a new materializer that extends ZenML's `PydanticMaterializer`
- [x] Keep only the visualization logic from the original materializer
- [x] Remove all manual JSON serialization/deserialization code

We'll create a materializer that extends the built-in PydanticMaterializer:

```python
from zenml.materializers import PydanticMaterializer
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
import os
from typing import Dict, Type, Any

class ResearchStateMaterializer(PydanticMaterializer):
    """Materializer for the ResearchState class with visualizations."""
    
    ASSOCIATED_TYPES = (ResearchState,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA
    
    def save_visualizations(
        self, data: ResearchState
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the ResearchState.
        
        Args:
            data: The ResearchState to visualize
            
        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization
        visualization_path = os.path.join(self.uri, "research_state.html")
        
        # Create HTML content based on current stage
        html_content = self._generate_visualization_html(data)
        
        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)
        
        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}
    
    def _generate_visualization_html(self, state: ResearchState) -> str:
        """Generate HTML visualization for the research state.
        
        Args:
            state: The ResearchState to visualize
            
        Returns:
            HTML string
        """
        # Copy the existing visualization generation logic
        # ...
```

### 3. Update Step Signatures and Methods
- [x] Modify step signatures to return separate state and HTML artifacts where useful
- [x] Update docstrings and type hints
- [x] Register materializers with ZenML steps

For key steps where HTML visualization is important (like final report):

```python
from typing import Annotated, Tuple
from zenml.types import HTMLString

@step(
    output_materializers={
        "state": ResearchStateMaterializer,
        "viz": None  # Default HTML materializer
    }
)
def final_report_step(
    state: ResearchState,
    llm_model: str = "gpt-4",
) -> Tuple[
    Annotated[ResearchState, "state"],
    Annotated[HTMLString, "viz"]
]:
    """Generate the final research report.
    
    Args:
        state: The research state with synthesized information
        llm_model: LLM model to use for report generation
        
    Returns:
        Tuple of (updated research state, HTML visualization)
    """
    # ... existing implementation ...
    
    # Generate HTML report
    report_html = generate_report_from_template(state, llm_model)
    
    # Update state
    state.final_report_html = report_html
    
    # Return both state and visualization
    return state, HTMLString(report_html)
```

For most steps, we can keep the single ResearchState return type:

```python
@step(output_materializers=ResearchStateMaterializer)
def process_sub_question_step(
    state: ResearchState,
    question_index: int,
    # ... other parameters
) -> Annotated[ResearchState, "output"]:
    """Process a single sub-question."""
    # ... implementation ...
    return sub_state
```

### 4. Update Import References and Fix Pipeline Structure
- [x] Update imports in all pipeline files
- [x] Fix pipeline structure to handle new output types
- [x] Update step references

Example of fixing a pipeline file:

```python
from typing import List, Optional

from zenml import pipeline
from zenml.types import HTMLString

# Update imports for the Pydantic models
from utils.pydantic_models import ResearchState

# Import your steps
from steps.query_decomposition_step import query_decomposition_step
from steps.process_sub_question_step import process_sub_question_step
from steps.merge_results_step import merge_results_step
from steps.final_report_step import final_report_step

@pipeline
def research_pipeline(
    query: str,
    # ... other parameters 
):
    """Pipeline for deep research with enhanced capabilities."""
    
    # Initialize research state
    initial_state = query_decomposition_step(query=query)
    
    # Process each sub-question in parallel
    sub_states = []
    for i in range(5):  # Support up to 5 sub-questions
        sub_state = process_sub_question_step(
            state=initial_state,
            question_index=i,
        )
        sub_states.append(sub_state)
    
    # Merge results
    merged_state = merge_results_step(
        initial_state=initial_state,
        sub_states=sub_states,
    )
    
    # Generate final report (returns tuple of state and HTML)
    final_state, report_html = final_report_step(
        state=merged_state,
    )
    
    return final_state, report_html
```

### 5. Clean Up Legacy Code and Testing
- [x] Remove old dataclass models once migration is complete
- [x] Remove manual serialization methods
- [x] Perform pipeline tests

## Detailed Implementation Steps

### Phase 1: Create Pydantic Models

1. **Setup and Preparation**
   - [x] Add Pydantic to requirements if not already there
   - [x] Create a new file `utils/pydantic_models.py` for new models

2. **Convert Simple Models**
   - [x] Implement `SearchResult` model
   - [x] Implement `ViewpointTension` model
   - [x] Test serialization/deserialization

3. **Implement Nested Models**
   - [x] Implement `SynthesizedInfo` model
   - [x] Implement `ViewpointAnalysis` model
   - [x] Implement `ReflectionMetadata` model
   - [x] Test nested serialization

4. **Create Main ResearchState**
   - [x] Implement `ResearchState` with all methods
   - [x] Configure model settings for mutability
   - [x] Test comprehensive serialization/deserialization

### Phase 2: Implement New Materializer

1. **Extract Visualization Logic**
   - [x] Copy HTML generation from current materializer
   - [x] Refactor as needed for Pydantic model access

2. **Create Extended Materializer**
   - [x] Create new `pydantic_materializer.py` file
   - [x] Implement class extending PydanticMaterializer
   - [x] Test basic saving/loading

3. **Test Visualization Integration**
   - [x] Test HTML generation with sample data
   - [x] Ensure compatibility with ZenML UI

### Phase 3: Update Step Signatures

1. **Identify Key Visualization Steps**
   - [x] Final report step
   - [x] Viewpoint analysis step
   - [x] Reflection step

2. **Update Step Decorators**
   - [x] Modify decorators to register materializers
   - [x] Update return type annotations

3. **Test Updated Steps**
   - [x] Test step execution
   - [x] Verify multiple outputs work correctly

### Phase 4: Refactor Pipeline Code

1. **Update Imports**
   - [x] Change import statements in all files
   - [x] Fix type annotations

2. **Pipeline Integration**
   - [x] Update pipeline to handle new return types
   - [x] Test full pipeline execution

3. **Final Cleanup**
   - [x] Remove old dataclass implementation
   - [x] Update documentation

## Migration Testing Strategy

1. **Unit Testing**
   - Test each model conversion with sample data
   - Verify serialization/deserialization works

2. **Integration Testing**
   - Test step execution with new models
   - Verify HTML artifacts appear in ZenML UI

3. **Pipeline Validation**
   - Run complete pipeline with test query
   - Compare results to pre-migration outputs

## Timeline

- Phase 1 (Models): 1-2 days
- Phase 2 (Materializer): 1 day
- Phase 3 (Step Signatures): 1-2 days
- Phase 4 (Cleanup): 1 day

Total estimated time: 4-6 days

## Conclusion

This migration will modernize the codebase by leveraging Pydantic's validation capabilities and ZenML's built-in support for Pydantic models. The result will be a more maintainable, type-safe implementation with improved visualization capabilities through first-class HTML artifacts.