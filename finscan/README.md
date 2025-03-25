# 🔍 FinScan: Financial Report Analysis Pipeline

Analyze complex financial reports with an orchestrated team of specialized AI agents. A system that processes financial documents, extracts insights, and validates data using ZenML, SmolAgents, and LangFuse.

## 🚀 Product Overview

FinScan is a specialized financial report analysis pipeline designed for analyzing lengthy financial documents. By combining ZenML, SmolAgents, and LangFuse, FinScan can:

- **Process Long Financial Reports**: Break down complex documents that would overwhelm a single LLM
- **Extract Key Financial Metrics**: Identify and extract important financial indicators
- **Provide Market Context**: Gather related market trends and analyst opinions
- **Analyze Competitors**: Compare with competitor reports for better context
- **Assess Risks**: Identify potential financial risks
- **Develop Strategic Insights**: Provide strategic recommendations

Built on ZenML's pipeline orchestration framework, FinScan delivers a reproducible solution with complete observability through LangFuse.

## 💡 How It Works

FinScan uses a structured pipeline approach with multiple specialized AI agents. The system follows these key steps:

1. **Document Preprocessing**
   - Financial reports are ingested and metadata is extracted
   - Extracted metadata is stored in the ZenML Store

2. **Agent Analysis**
   - **Metrics Agent**: Identifies and extracts financial metrics
   - **Context Agent**: Understands the broader financial landscape
   - **Competitor Agent**: Conducts comparisons with competitor reports
   - **Risk Agent**: Assesses financial risks
   - **Strategy Agent**: Provides strategic recommendations

3. **Synthesis and Validation**
   - **Consistency Checker**: Ensures data accuracy
   - **Gap Analysis**: Detects missing or conflicting information
   - **Synthesis**: Compiles the final structured financial insights

4. **Evaluation Pipeline**
   - Validated insights are displayed in a dashboard

## 🎯 Use Cases

The primary use case for FinScan is financial report analysis, where:

- Single LLMs struggle to read long reports and extract key information
- Multiple AI agents handle different tasks, improving overall results
- Vertical AI agents automate real, high-value work with deep domain expertise

## 🔧 Getting Started

### Prerequisites

- Python 3.9+
- ZenML installed and configured
- API keys for required services:
  - OpenAI
  - Langfuse
  - HuggingFace (optional)
  - SearchAPI

### Quick Setup

1. Install requirements:
   ```bash
   pip install zenml openai pandas smolagents[telemetry] opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents langchain-core langchain-community
   ```

2. Configure API keys:
   ```bash
   # Create an .env file with your API keys
   OPENAI_API_KEY=your_openai_api_key
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   HF_TOKEN=your_huggingface_token
   SEARCHAPI_API_KEY=your_searchapi_key
   ```

3. Login to ZenML Pro:
   ```bash
   # Log in to ZenML
   zenml login
   ```

## 📊 Observability

FinScan emphasizes the importance of observability for AI agents:

- LangFuse provides real-time tracking of agent decisions
- Tracks which tools an agent calls
- Shows what data is retrieved
- Demonstrates how decisions evolve
- Enables debugging of agent behavior

## 📁 Technical Components

The main components of FinScan include:

- **ZenML**: Orchestrates the pipeline, ensuring smooth data flow and reproducibility
- **SmolAgents**: Keeps the AI logic simple, modular, and extensible
- **LangFuse**: Provides observability for agent behavior

## 🗂️ Dataset

FinScan uses the FINDSum dataset:

- 21,125 annual reports from 3,794 companies
- Divided into two subsets:
  - FINDSum-ROO: Summarizes a company's results of operations
  - FINDSum-Liquidity: Focuses on liquidity and capital resources

This dataset is designed for financial document summarization and integrates numerical values from tables to improve summary informativeness.

## 📚 Additional Resources

For more information on the tools used in this project:

- [ZenML Documentation](https://docs.zenml.io/)
- [LangFuse Documentation](https://langfuse.com/docs)
- [SmolAgents GitHub Repository](https://github.com/huggingface/smolmodels)
