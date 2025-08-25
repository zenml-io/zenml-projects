# 🧪 QualityFlow: AI-Powered Test Generation Pipeline

A streamlined MLOps pipeline for **automated test generation** using ZenML and LLMs. Generate comprehensive unit tests for your codebase, compare different approaches, and get detailed coverage analysis.

## 🚀 Product Overview

QualityFlow demonstrates how to build production-ready MLOps workflows for automated test generation using Large Language Models. Built with ZenML, it provides a simple yet powerful pipeline for generating and evaluating AI-generated tests.

**Focus**: **LLM-Powered Test Generation** and **Coverage Analysis**.

### Key Features

- **Real LLM Integration**: OpenAI and Anthropic providers for intelligent test generation
- **Smart File Selection**: Configurable strategies to focus on files that need testing
- **Baseline Comparison**: Compare LLM-generated tests vs heuristic baseline tests
- **Coverage Analysis**: Real coverage metrics with detailed reporting
- **Speed Controls**: `max_files` parameters to control pipeline execution time
- **Containerized Ready**: Uses ZenML Path artifacts for remote execution
- **Cost Tracking**: Token usage and cost estimation with metadata logging

## 💡 How It Works

### ✈️ Pipeline Architecture

QualityFlow consists of a single, focused pipeline:

#### Generate & Evaluate Pipeline

The main pipeline handles the complete test generation workflow:

1. **Source Selection** - Specify repository and target files
2. **Code Fetching** - Clone and materialize workspace 
3. **Code Analysis** - Select files for testing (with max_files limit)
4. **LLM Test Generation** - Generate tests using OpenAI/Anthropic/fake providers
5. **Baseline Generation** - Create simple heuristic tests for comparison
6. **Test Execution** - Run both test suites with coverage analysis
7. **Report Generation** - Compare results and generate markdown reports

### 🔧 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Git Repo      │    │  LLM Providers  │    │   Test Reports  │
│                 │    │                 │    │                 │
│  src/**/*.py    │────│▶ OpenAI/Claude  │────│▶ Coverage       │
│  target files   │    │  Fake (testing) │    │  Comparisons    │
│                 │    │                 │    │  Cost Tracking  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                                             ▲
         ▼                                             │
┌─────────────────────────────────────────────────────────────────┐
│                   QualityFlow Pipeline                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Generate & Evaluate                      │   │
│  │                                                         │   │
│  │ 1. Select Input    → 2. Fetch Source    → 3. Analyze   │   │
│  │ 4. Generate (LLM)  → 5. Generate (Base) → 6. Run Tests │   │
│  │ 7. Run Tests       → 8. Report & Compare               │   │
│  │                                                         │   │
│  │ Features: max_files control, Path artifacts, metadata  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

Get QualityFlow running in 3 simple steps:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Optional: Set up OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```
*Skip this step to use the fake provider for testing*

### 3. Run the Pipeline
```bash
python run.py
```

**That's it!** The pipeline will automatically:
- Clone a sample repository (requests library by default)
- Analyze Python files and select test candidates
- Generate tests using LLM or fake provider
- Run tests and measure coverage
- Create a detailed comparison report

### What Happens Next?

- Check the ZenML dashboard to see pipeline results
- View generated test files and coverage reports
- Compare LLM vs baseline test approaches
- Experiment with different configurations

## ⚙️ Configuration

### Key Parameters

You can customize the pipeline behavior by editing `configs/experiment.default.yaml`:

```yaml
# Control execution speed
steps:
  analyze_code:
    parameters:
      max_files: 3  # Limit files to analyze (faster execution)
  
  gen_tests_agent:
    parameters:
      provider: "openai"  # openai | anthropic | fake
      model: "gpt-4o-mini"
      max_files: 2        # Limit files for test generation
      max_tests_per_file: 3

  gen_tests_baseline:
    parameters:
      max_files: 2        # Match agent for fair comparison
```

### Pipeline Options

```bash
# Use fake provider (no API key needed)
python run.py  # Uses config defaults

# Force fresh execution (no caching) 
python run.py --no-cache

# Use different config
python run.py --config configs/experiment.strict.yaml
```

## 🔬 Advanced Usage

### Different Target Repositories

Edit the config to point to your own repository:

```yaml
steps:
  select_input:
    parameters:
      repo_url: "https://github.com/your-org/your-repo.git"
      ref: "main"
      target_glob: "src/**/*.py"  # Adjust path pattern
```

### Custom Prompts

Create new Jinja2 templates in `prompts/`:

```jinja2
# prompts/custom_test_v3.jinja

Generate {{ max_tests }} tests for:
{{ file_path }} (complexity: {{ complexity_score }})

Source:
```python
{{ source_code }}
```

Requirements:
- Use pytest fixtures
- Include edge cases
- Mock external dependencies
```

### A/B Testing Experiments

Compare different configurations by running with different config files:

```bash
# Compare prompt versions
python run.py --config configs/experiment.default.yaml
python run.py --config configs/experiment.strict.yaml

# Compare results in ZenML dashboard:
# - Coverage metrics
# - Test quality scores  
# - Token usage and cost
```

### Production Deployment

Set up ZenML stack for cloud deployment:

```bash
# Example: AWS EKS stack
zenml artifact-store register s3_store --flavor=s3 --path=s3://your-bucket
zenml container-registry register ecr_registry --flavor=aws --uri=your-account.dkr.ecr.region.amazonaws.com
zenml orchestrator register k8s_orchestrator --flavor=kubernetes --kubernetes_context=your-eks-context

zenml stack register production_stack \
  -a s3_store -c ecr_registry -o k8s_orchestrator --set
```

### Scheduled Execution

For automated runs, set up scheduled execution using your preferred orchestration tool or ZenML's scheduling features.

## 🏗️ Project Structure

```
qualityflow/
├── README.md
├── requirements.txt
│
├── configs/                          # Pipeline configurations
│   ├── experiment.default.yaml       # Standard experiment settings
│   └── experiment.strict.yaml        # High-quality gates
│
├── pipelines/                        # Pipeline definitions
│   └── generate_and_evaluate.py      # Main pipeline
│
├── steps/                            # Pipeline steps
│   ├── select_input.py               # Source specification
│   ├── fetch_source.py               # Repository fetching  
│   ├── analyze_code.py               # Code analysis & selection
│   ├── gen_tests_agent.py            # LLM test generation
│   ├── gen_tests_baseline.py         # Heuristic test generation
│   ├── run_tests.py                  # Test execution & coverage
│   ├── evaluate_coverage.py          # Metrics evaluation
│   └── report.py                     # Report generation
│
├── prompts/                          # Jinja2 prompt templates
│   ├── unit_test_v1.jinja           # Standard test generation
│   └── unit_test_strict_v2.jinja    # Comprehensive test generation
│
├── examples/                         # Demo code for testing
│   └── toy_lib/                     # Sample library
│       ├── calculator.py
│       └── string_utils.py
│
└── run.py                           # Main entry point
```

### Key Components

- **Pipeline Steps**: Modular, reusable components with clear interfaces
- **Prompt Templates**: Jinja2 templates for LLM test generation  
- **Configuration**: YAML-driven experiment settings
- **Test Generation**: Both LLM-based and heuristic approaches for comparison

## 🚀 Production Deployment

### ZenML Cloud Stack Setup

For production deployment with ZenML Cloud:

```bash
# Connect to ZenML Cloud
zenml connect --url https://your-org.zenml.cloud

# Register cloud stack components
zenml artifact-store register cloud_store --flavor=s3 --path=s3://qualityflow-artifacts
zenml orchestrator register cloud_k8s --flavor=kubernetes --kubernetes_context=prod-cluster

zenml stack register production \
  -a cloud_store -o cloud_k8s --set
```

### Scheduled Execution

Set up automated regression testing using ZenML's scheduling capabilities or your preferred orchestration platform.

## 🤝 Contributing

QualityFlow follows ZenML best practices and is designed to be extended:

1. **Add New LLM Providers**: Extend `gen_tests_agent.py` with new provider integrations
2. **Custom Materializers**: Create materializers for new artifact types
3. **Additional Metrics**: Expand evaluation capabilities with new quality metrics
4. **Selection Strategies**: Add new code selection algorithms

## 📝 Next Steps

After running QualityFlow successfully:

1. **Explore ZenML Dashboard**: View pipeline runs, artifacts, and model registry
2. **Experiment with Prompts**: Try different test generation strategies
3. **Add Real Codebases**: Replace toy examples with your production code
4. **Deploy to Production**: Use cloud orchestration for scale
5. **Set Up Monitoring**: Configure alerts for regression detection

## 🆘 Troubleshooting

### Common Issues

**LLM API Errors**:
- Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variables
- Use `provider: "fake"` for development without API keys

**Test Execution Failures**:
- Ensure pytest and coverage tools are installed
- Check that workspace has proper Python path setup

### Debug Mode

Run with debug logging:

```bash
export ZENML_LOGGING_VERBOSITY=DEBUG
python run.py --config configs/experiment.default.yaml
```

## 📚 Resources

- [ZenML Documentation](https://docs.zenml.io/)
- [Model Control Plane](https://docs.zenml.io/user-guide/model-control-plane)
- [Kubernetes Orchestrator](https://docs.zenml.io/stacks/stack-components/orchestrators/kubernetes)

---

Built with ❤️ using [ZenML](https://zenml.io) - *The MLOps Framework for Production AI*