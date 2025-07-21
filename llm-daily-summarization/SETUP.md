# LLM Summarization Pipeline - Setup Guide

This guide walks you through setting up and running the LLM Summarization Pipeline from scratch.

## Prerequisites

- Python 3.9+ 
- Google Cloud Platform account with Vertex AI enabled
- Langfuse account (free tier available)
- Discord/Slack API access (optional, for real data)

## Installation

### 1. Clone and Navigate to Project

```bash
cd /path/to/zenml-projects/llm-summarization-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Basic Setup

```bash
python test_basic.py
```

You should see:
```
🎉 All basic tests passed! Core functionality is working correctly.
```

## Configuration

### 1. Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```bash
# === LLM Provider Configuration ===
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# === Observability ===
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."

# === Data Sources (Optional) ===
DISCORD_BOT_TOKEN="your-discord-bot-token"
SLACK_BOT_TOKEN="xoxb-your-slack-bot-token"

# === Output Destinations (Optional) ===
NOTION_TOKEN="secret_your-notion-integration-token"
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

### 2. Google Cloud Setup

1. Create a GCP project with Vertex AI enabled
2. Create a service account with Vertex AI permissions
3. Download the service account key JSON file
4. Set `GOOGLE_APPLICATION_CREDENTIALS` to the file path

### 3. Langfuse Setup

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a new project
3. Copy the public and secret keys to your `.env`

### 4. ZenML Configuration (Optional)

Initialize ZenML for production usage:

```bash
zenml init
zenml stack register llm-ai-agents-stack \
  --orchestrator default \
  --artifact-store default
```

## Running the Pipeline

### Option 1: Mock Data (Recommended for Testing)

Run with sample data (no external API calls needed):

```bash
python run.py
```

This uses the sample conversations in `data/sample_conversations.json`.

### Option 2: Real Data Sources

To use real Discord/Slack data:

1. Set up Discord/Slack API access
2. Configure tokens in `.env`
3. Update `run.py` to use real data ingestion:

```python
# In run.py, change:
from src.steps.mock_data_ingestion import mock_chat_data_ingestion_step
# to:
from src.steps.data_ingestion import chat_data_ingestion_step
```

## Pipeline Components

### 1. Data Flow

```
Discord/Slack → Ingestion → Preprocessing → LangGraph → Output → Evaluation
     ↓             ↓           ↓           ↓         ↓        ↓
Sample Data → Mock Step → Text Cleaning → Agents → Slack → Metrics
```

### 2. Key Components

- **Data Ingestion**: Fetches chat messages from Discord/Slack APIs
- **Preprocessing**: Cleans and filters messages 
- **LangGraph Agents**: Multi-agent workflow with Summarizer and Task Extractor
- **Output Distribution**: Delivers results to Slack, Notion, GitHub
- **Evaluation**: Tracks quality metrics and costs

### 3. Monitoring

- **Langfuse**: Traces all LLM calls and agent decisions
- **ZenML**: Tracks pipeline runs and artifacts
- **Evaluation**: Automatic quality scoring and cost tracking

## Customization

### Adding New Data Sources

1. Create a new client in `src/steps/data_ingestion.py`
2. Follow the pattern of `DiscordClient` or `SlackClient`
3. Update the configuration to include your source

### Adding New Output Targets

1. Create a new deliverer in `src/steps/output_distribution.py`
2. Follow the pattern of `NotionDeliverer` or `SlackDeliverer`
3. Update the configuration to include your target

### Modifying Agents

1. Update prompts in `src/agents/summarizer_agent.py` or `src/agents/task_extractor_agent.py`
2. Add new agents by creating similar classes
3. Update the LangGraph workflow in `src/steps/langgraph_processing.py`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and you're in the correct directory
2. **Authentication Errors**: Verify your API keys and credentials are correct
3. **Model Access**: Ensure your GCP project has Vertex AI enabled
4. **ZenML Connection**: For ZenML errors, try running in mock mode first

### Debugging

1. **Enable Debug Logging**:
   ```bash
   export ZENML_LOGGING_VERBOSITY=DEBUG
   ```

2. **Check Langfuse Traces**: Visit your Langfuse dashboard to see LLM call traces

3. **Run Basic Tests**:
   ```bash
   python test_basic.py
   ```

## Development

### Running Tests

```bash
# Basic functionality tests
python test_basic.py

# Full test suite (requires dependencies)
python -m pytest tests/ -v
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Adding Features

1. Create feature branch
2. Add your changes following existing patterns
3. Add tests in `tests/` directory
4. Update documentation as needed

## Production Deployment

### 1. Environment Setup

- Use proper secret management for API keys
- Set up monitoring and alerting
- Configure proper logging

### 2. Scheduling

Set up daily runs using:
- Cron jobs
- ZenML schedulers
- Cloud Functions/Lambda

### 3. Scaling

- Use ZenML cloud orchestrators for scaling
- Configure resource limits for LLM calls
- Set up cost monitoring and budgets

## Cost Management

### Monitoring Costs

The pipeline automatically tracks:
- Token usage per run
- Estimated costs based on model pricing
- API call counts

### Cost Optimization

1. **Adjust Model Parameters**:
   - Reduce `max_tokens` for shorter outputs
   - Use lower temperature for more deterministic results

2. **Filter Data**:
   - Limit days_back for data ingestion
   - Filter out low-value conversations

3. **Batch Processing**:
   - Process multiple conversations together
   - Use caching for repeated content

## Support

- Check the troubleshooting section above
- Review Langfuse traces for LLM issues
- Examine ZenML logs for pipeline issues
- Test with mock data first before using real APIs