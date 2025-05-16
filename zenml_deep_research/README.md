# 🔍 ZenML Deep Research Agent

A production-ready MLOps pipeline for conducting deep, comprehensive research on any topic using LLMs and web search capabilities.

<div align="center">
  <p><em>ZenML pipeline for advanced research and report generation</em></p>
</div>

## 🎯 Overview

The ZenML Deep Research Agent is a scalable, modular pipeline that automates in-depth research on any topic. It:

- Creates a structured outline based on your research query
- Researches each section through targeted web searches and LLM analysis
- Iteratively refines content through reflection cycles
- Produces a comprehensive, well-formatted research report

This project transforms exploratory notebook-based research into a production-grade, reproducible, and transparent process using the ZenML MLOps framework.

## 🚀 Pipeline Architecture

The pipeline implements a complete research workflow with the following steps:

1. **Configuration Loading**: Load settings from YAML and environment variables
2. **Report Structure Generation**: Create a structured outline for the research
3. **Paragraph Research**: For each paragraph in the outline:
   - Generate optimal search queries
   - Retrieve and analyze web content
   - Synthesize findings into coherent paragraphs
   - Iteratively refine through multiple reflection cycles
4. **Report Formatting**: Compile researched content into a polished, formatted report

## 💡 Under the Hood

- **LLM Integration**: Uses SambaNova API with reasoning and instruction-tuned models
- **Web Research**: Utilizes Tavily API for targeted internet searches
- **ZenML Orchestration**: Manages pipeline flow, artifacts, and caching
- **Reproducibility**: Track every step, parameter, and output via ZenML

## 🛠️ Getting Started

### Prerequisites

- Python 3.9+
- ZenML installed and configured
- SambaNova API key
- Tavily API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd zenml_deep_research

# Install dependencies
pip install -r requirements.txt

# Set up API keys
export SAMBANOVA_API_KEY=your_sambanova_key
export TAVILY_API_KEY=your_tavily_key

# Initialize ZenML (if needed)
zenml init
```

### Running the Pipeline

#### Basic Usage

```bash
python run.py --query "Explain the impact of quantum computing on cryptography"
```

#### Using Different Configurations

```bash
python run.py --query "History of artificial intelligence" --config configs/custom_config.yaml
```

#### Saving the Report to a File

```bash
python run.py --query "Climate change adaptation strategies" --output report.html
```

### Advanced Options

```bash
# Enable debug logging
python run.py --query "Your query" --debug

# Disable caching for a fresh run
python run.py --query "Your query" --no-cache

# Specify a log file
python run.py --query "Your query" --log-file research.log
```

## 📁 Project Structure

```
zenml_deep_research/
├── configs/             # Configuration files
│   ├── __init__.py
│   └── research_config.yaml
├── pipelines/           # ZenML pipeline definitions
│   ├── __init__.py
│   └── research_pipeline.py
├── steps/               # ZenML pipeline steps
│   ├── __init__.py
│   ├── load_config.py
│   ├── report_structure_step.py
│   ├── paragraph_research_step.py
│   └── report_formatting_step.py
├── utils/               # Utility functions and helpers
│   ├── __init__.py
│   ├── data_models.py
│   └── helper_functions.py
├── __init__.py
├── requirements.txt     # Project dependencies
├── logging_config.py    # Logging configuration
├── README.md            # Project documentation
└── run.py               # Main script to run the pipeline
```

## 🔧 Customization

The project supports two levels of customization:

### 1. Step Parameters

You can customize the research behavior directly through command-line parameters:

```bash
# Run with more reflection cycles
python run.py --query "Your query" --num-reflections 3

# Adjust search parameters (available as step parameters)
python run.py --query "Your query" --num-reflections 4
```

Each step has its own parameters with sensible defaults that can be customized by modifying the step definitions.

### 2. Pipeline Configuration

For pipeline-level settings, modify the configuration file:

```yaml
# configs/pipeline_config.yaml

# Pipeline settings
pipeline:
  name: "deep_research_pipeline"
  enable_cache: true
  
# Environment settings
environment:
  docker:
    requirements:
      - openai>=1.0.0
      - tavily-python>=0.2.8
      - PyYAML>=6.0

# Resource configuration
resources:
  cpu: 2  # Increase for faster processing
  memory: "8Gi"  # Increase for larger research tasks
  
# Maximum execution time in seconds
timeout: 7200  # Increase for more complex research
```

To use a custom configuration file:

```bash
python run.py --query "Your query" --config configs/custom_pipeline.yaml
```

## 📈 Example Use Cases

- **Academic Research**: Rapidly generate preliminary research on academic topics
- **Business Intelligence**: Stay informed on industry trends and competitive landscape
- **Content Creation**: Develop well-researched content for articles, blogs, or reports
- **Decision Support**: Gather comprehensive information for informed decision-making

## 🔄 Integration Possibilities

This pipeline can integrate with:

- **Document Storage**: Save reports to database or document management systems
- **Web Applications**: Power research functionality in web interfaces
- **Alerting Systems**: Schedule research on key topics and receive regular reports
- **Other ZenML Pipelines**: Chain with downstream analysis or processing

## 📄 License

This project is licensed under the Apache License 2.0.
