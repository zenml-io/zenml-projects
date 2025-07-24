# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Testing
```bash
# Install dependencies (use `uv` if available)
pip install -r requirements.txt

# Basic functionality test (no external dependencies)
python test_basic.py

# Run the main pipeline with mock data
python run.py

# Code formatting
bash ../scripts/format.sh
```

### Pipeline Execution
```bash
# Run with mock data (default - no API keys required)
python run.py

# Enable debug logging
export ZENML_LOGGING_VERBOSITY=DEBUG
python run.py
```

## Architecture Overview

This is a **production-ready LLMOps pipeline** that processes team conversations and generates summaries and task lists. The architecture follows modern AI operations best practices:

### Core Technology Stack
- **ZenML**: Pipeline orchestration and artifact management
- **LangGraph**: Multi-agent workflow coordination and state management
- **Vertex AI**: Gemini 2.5 Flash for LLM processing
- **Langfuse**: Complete LLM observability and tracing
- **Multiple Integrations**: Discord, Slack, Notion, GitHub APIs

### Pipeline Flow
```
Data Sources → Ingestion → Preprocessing → LangGraph Agents → Output Distribution → Evaluation
    ↓             ↓           ↓              ↓                    ↓                ↓
Discord/Slack → Mock/Real → Text Clean → Summarizer/Tasks → Slack/Notion → Metrics
```

### Key Components

1. **ZenML Pipeline Steps** (`src/steps/`):
   - `data_ingestion.py` - Discord/Slack API clients
   - `mock_data_ingestion.py` - Testing with sample data (default)
   - `preprocessing.py` - Text cleaning and filtering
   - `langgraph_processing.py` - Multi-agent orchestration hub
   - `output_distribution.py` - Multi-platform delivery
   - `evaluation.py` - Quality metrics and monitoring

2. **LangGraph Agents** (`src/agents/`):
   - `summarizer_agent.py` - Creates conversation summaries
   - `task_extractor_agent.py` - Extracts action items and assignments
   - Both agents use `@observe` decorators for Langfuse tracing

3. **State Management**:
   - `AgentState` (TypedDict) passed between agents
   - Includes conversations, summaries, tasks, metadata, and usage stats
   - LangGraph coordinates agent execution and state transitions

### Configuration and Environment

The pipeline uses environment variables for configuration:
- **LLM**: `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`
- **Observability**: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
- **Data Sources**: `DISCORD_BOT_TOKEN`, `SLACK_BOT_TOKEN`
- **Output**: `NOTION_TOKEN`, `SLACK_WEBHOOK_URL`

Default model configuration uses Gemini 2.5 Flash with conservative settings (temperature=0.1, 4000 max tokens).

### Testing and Debugging

The pipeline supports two execution modes:
1. **Mock Mode** (default): Uses `data/sample_conversations.json` - no API keys required
2. **Production Mode**: Requires full environment setup and API credentials

Always start with `python test_basic.py` to verify core functionality before attempting full runs.

### Data Models (`src/utils/models.py`)

Uses Pydantic models for type safety:
- `ConversationData` - Raw chat message structure
- `CleanedConversationData` - Processed conversation data
- `Summary` - Generated summary with metadata
- `TaskItem` - Extracted tasks with assignments
- `ProcessedData` - Complete pipeline output

### Observability

Every LLM call and agent decision is traced through Langfuse using `@observe` decorators. The pipeline automatically tracks:
- Token usage and estimated costs
- Processing time and success rates
- Quality metrics and evaluation scores
- Complete agent workflow execution traces
