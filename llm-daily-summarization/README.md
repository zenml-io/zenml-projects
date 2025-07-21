# LLM Summarization Pipeline

**Production-ready daily chat summarization pipeline using ZenML, LangGraph, Vertex AI, and Langfuse**

## 🎯 Overview

This project demonstrates a complete LLMOps pipeline that automatically processes team conversations and generates actionable summaries and task lists. Built with modern AI operations best practices, it showcases:

- **End-to-End Automation**: From chat ingestion to team delivery
- **Multi-Agent Coordination**: LangGraph orchestrates specialized AI agents
- **Production Observability**: Complete LLM call tracing and evaluation
- **Modular Architecture**: Easy to extend and customize for different use cases

## 🏗️ Architecture

```mermaid
graph LR
    A[Discord/Slack] → B[Data Ingestion]
    B → C[Text Preprocessing] 
    C → D[LangGraph Agents]
    D → E[Summarizer Agent]
    D → F[Task Extractor Agent]
    E → G[Output Distribution]
    F → G
    G → H[Slack/Notion/GitHub]
    
    I[Langfuse] ← D
    J[ZenML] ← B
    J ← C
    J ← D
    J ← G
```

**Core Components:**
- **ZenML Pipeline**: Orchestration and artifact management
- **LangGraph Workflow**: Multi-agent coordination and state management  
- **Vertex AI**: Gemini 2.5 Flash for high-quality, cost-effective processing
- **Langfuse**: Complete LLM observability and evaluation
- **Multiple Integrations**: Discord, Slack, Notion, GitHub APIs

## 📁 Project Structure

```
llm-summarization-pipeline/
├── src/
│   ├── steps/              # ZenML pipeline steps
│   │   ├── data_ingestion.py        # Discord/Slack API clients
│   │   ├── mock_data_ingestion.py   # Testing with sample data
│   │   ├── preprocessing.py         # Text cleaning and filtering
│   │   ├── langgraph_processing.py  # Multi-agent orchestration
│   │   ├── output_distribution.py   # Multi-platform delivery
│   │   └── evaluation.py            # Quality metrics and monitoring
│   ├── agents/             # LangGraph agent definitions
│   │   ├── summarizer_agent.py      # Conversation summarization
│   │   └── task_extractor_agent.py  # Action item identification
│   ├── utils/              # Shared utilities
│   │   └── models.py                # Pydantic data models
│   └── materializers/      # Custom ZenML serializers
├── configs/                # Configuration files
│   └── stack_config.yaml           # ZenML stack definition
├── data/                   # Sample conversations for testing
│   └── sample_conversations.json
├── tests/                  # Test suite
├── .env.example           # Environment variables template
├── requirements.txt       # Dependencies
├── SETUP.md              # Detailed setup instructions
└── run.py                # Main pipeline entry point
```

## 🚀 Quick Start

### Option 1: Test with Sample Data (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test basic functionality
python test_basic.py

# 3. Run with sample data (no API keys needed)
python run.py
```

### Option 2: Full Setup with Real APIs

See [SETUP.md](SETUP.md) for complete configuration instructions including:
- Google Cloud / Vertex AI setup
- Langfuse account configuration  
- Discord/Slack API access
- Production deployment guidelines

## ✨ Key Features

### 🤖 Multi-Agent Processing
- **Summarizer Agent**: Creates comprehensive conversation summaries
- **Task Extractor Agent**: Identifies action items and assignments
- **Quality Checker**: Validates output quality and consistency
- **LangGraph Coordination**: Manages agent workflows and state

### 📊 Complete Observability
- **LLM Call Tracing**: Every model interaction logged in Langfuse
- **Cost Tracking**: Real-time token usage and cost estimation
- **Quality Metrics**: Automated evaluation of summary quality
- **Performance Monitoring**: Processing time and success rate tracking

### 🔧 Production Ready
- **Error Handling**: Robust retry logic and graceful degradation
- **Scalable Architecture**: ZenML enables cloud orchestration
- **Configuration Management**: Environment-based config with secrets
- **Testing Suite**: Comprehensive tests for all components

### 🎛️ Flexible Integration
- **Data Sources**: Discord, Slack, custom chat platforms
- **Output Targets**: Slack, Notion, GitHub Issues, Discord
- **Model Support**: Easy to swap between different LLM providers
- **Custom Agents**: Extensible agent framework

## 📈 Example Output

### Summary Example
```
📝 Daily Engineering Team Summary

Key discussions focused on authentication feature implementation and user research findings. 
The team reviewed PR feedback and established next steps for security improvements.

Key Points:
• Authentication feature ready for review with focus on token validation
• User research completed with 15 participants - navigation concerns identified  
• Security review scheduled for Wednesday
• API documentation updates planned post-merge

Participants: Alice, Bob, Charlie, Diana, Eve, Frank
Topics: Authentication, Security, User Research, Documentation
```

### Task Extraction Example
```
✅ Action Items (3 tasks)

🔥 Security review of authentication feature
   Review token validation logic in auth_utils.py with focus on edge cases
   👤 Assigned to: Charlie
   📅 Due: Wednesday

⚡ Update API documentation  
   Document new authentication endpoints after feature merge
   👤 Assigned to: Bob

📝 Prepare user research report
   Compile findings and suggestions from 15 user interviews
   👤 Assigned to: Diana
   📅 Due: Friday
```

## 🔍 Monitoring & Evaluation

### Langfuse Dashboard
- **Trace Visualization**: See complete agent workflow execution
- **Token Analytics**: Track usage patterns and costs over time
- **Quality Scores**: Monitor summary and task extraction accuracy
- **Error Analysis**: Debug failed runs and model issues

### ZenML Integration  
- **Pipeline Versioning**: Track changes to models and prompts
- **Artifact Management**: Store and version all pipeline outputs
- **Experiment Tracking**: Compare different agent configurations
- **Deployment Management**: Handle staging and production environments

## 🛠️ Development

### Running Tests
```bash
# Basic functionality (no external dependencies)
python test_basic.py

# Full test suite (requires setup)
python -m pytest tests/ -v
```

### Adding Custom Agents
```python
# Create new agent following existing patterns
class MyCustomAgent:
    def __init__(self, model_config):
        self.llm = ChatVertexAI(**model_config)
    
    @observe(as_type="generation")  # Langfuse tracing
    def process(self, data):
        # Your agent logic here
        pass
```

### Extending Output Targets
```python
# Add new delivery platform
class CustomDeliverer:
    @observe(as_type="span")
    def deliver_summary(self, summary_data):
        # Integration with your platform
        pass
```

## 📊 Performance & Costs

### Typical Usage (per day)
- **Processing Time**: 2-5 minutes for 100 messages
- **Token Usage**: ~2,000 tokens (Gemini 2.5 Flash)
- **Estimated Cost**: $0.01-$0.05 per day
- **API Calls**: 3-5 LLM calls per conversation

### Optimization Tips
- Adjust `max_tokens` based on summary length needs
- Use conversation filtering to process only relevant channels
- Batch multiple conversations for efficiency
- Monitor costs through Langfuse analytics

## 🤝 Contributing

This project serves as a reference implementation for LLMOps best practices. Contributions welcome for:
- Additional data source integrations
- New output platform support  
- Enhanced evaluation metrics
- Performance optimizations

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with:** ZenML • LangGraph • Vertex AI • Langfuse • Python