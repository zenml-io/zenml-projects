# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Pipeline
```bash
# Main entry point with different research modes
python run.py --mode rapid      # Quick research (5 sub-questions, 2 results each)
python run.py --mode balanced   # Standard research (10 sub-questions, 3 results each) 
python run.py --mode deep       # Comprehensive research (15 sub-questions, 5 results each)

# Custom configurations
python run.py --config configs/custom.yaml
python run.py --query "Research topic" --max-sub-questions 15
python run.py --require-approval --search-provider both

# Tracking provider options
python run.py --tracking-provider weave     # Use Weave tracking (default)
python run.py --tracking-provider langfuse  # Use Langfuse tracking
python run.py --tracking-provider none      # Disable tracking
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_models.py
pytest tests/test_approval_utils.py
```

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Debug mode with logging
python run.py --debug --log-file research.log
```

## Architecture

This is a **ZenML MLOps pipeline** for automated research using LLMs and web search. The architecture uses a **parallel fan-out/fan-in pattern**:

1. **Query Decomposition** → Break main query into sub-questions
2. **Parallel Processing** → Process sub-questions concurrently 
3. **Cross-Viewpoint Analysis** → Analyze different perspectives
4. **MCP Integration** → Anthropic's Model Context Protocol searches
5. **Final Report** → Generate comprehensive HTML report

### Key Directories
- `configs/` - Pipeline configurations (6 research modes)
- `steps/` - Individual pipeline steps (11 modular steps)
- `materializers/` - Custom ZenML materializers (9 types)
- `utils/` - Utility functions and helpers
- `pipelines/` - Pipeline definitions
- `tests/` - Test suite with pytest

### Data Models
The codebase uses **Pydantic models** extensively for data validation and serialization. All pipeline artifacts are strongly typed and versioned through ZenML.

## Environment Variables
Required API keys for full functionality:
- `SAMBANOVA_API_KEY` - Primary LLM provider
- `TAVILY_API_KEY` - Web search provider
- `EXA_API_KEY` - Alternative search + MCP integration
- `ANTHROPIC_API_KEY` - MCP integration

**Tracking Provider APIs (choose one):**
- `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY` - Langfuse tracking
- `WANDB_API_KEY` - Weave/Wandb tracking

## Pipeline Modes
The pipeline supports 6 different research modes via YAML configs:
- `rapid`, `balanced`, `deep` - Preset modes with different depth levels
- `enhanced`, `quick`, `with_approval` - Specialized configurations

Each mode controls parameters like `max_sub_questions`, `num_results_per_search`, and `max_additional_searches`.

## Key Features
- **Parallel processing** for faster research completion
- **Human-in-the-loop** approval workflows (optional)
- **Multi-provider support** for LLMs (via LiteLLM) and search (Tavily + Exa)
- **Flexible experiment tracking** - supports Langfuse, Weave, or no tracking
- **Comprehensive tracking** of costs, tokens, and performance metrics
- **MCP integration** for enhanced searches using Anthropic's Model Context Protocol
- **Artifact versioning** and caching through ZenML

## Testing Strategy
Tests are organized into 5 modules covering:
- Pydantic model validation
- Approval utilities
- Artifact models  
- Pipeline step functionality
- Integration testing

The test suite uses pytest with a custom conftest.py for path setup.