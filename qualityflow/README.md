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

## 📦 Quick Start

### Prerequisites

- Python 3.9+
- ZenML installed (`pip install zenml`)
- Git
- OpenAI API key (optional, can use fake provider)

### Setup

```bash
pip install -r requirements.txt
```

2. **Set up OpenAI (optional)**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Run the pipeline**:
```bash
python run.py
```

That's it! The pipeline will:
- Clone the configured repository (default: requests library)
- Analyze Python files and select candidates
- Generate tests using OpenAI (or fake provider if no API key)
- Run tests and measure coverage
- Generate a comprehensive report comparing approaches

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

Use run templates for systematic comparisons:

```bash
# Compare prompt versions
python scripts/run_experiment.py --config configs/experiment.default.yaml
python scripts/run_experiment.py --config configs/experiment.strict.yaml

# Compare in ZenML dashboard:
# - Coverage metrics
# - Test quality scores  
# - Token usage and cost
# - Promotion decisions
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

### Scheduled Regression

Register batch regression for daily execution:

```bash
python scripts/run_batch.py --config configs/schedule.batch.yaml --schedule
```

## 🏗️ Project Structure

```
qualityflow/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── zenml.yaml
│
├── configs/                          # Pipeline configurations
│   ├── experiment.default.yaml       # Standard experiment settings
│   ├── experiment.strict.yaml        # High-quality gates
│   └── schedule.batch.yaml           # Batch regression schedule
│
├── domain/                           # Core data models
│   ├── schema.py                     # Pydantic models
│   └── stages.py                     # Deployment stages
│
├── pipelines/                        # Pipeline definitions
│   ├── generate_and_evaluate.py      # Experiment pipeline
│   └── batch_regression.py           # Scheduled regression
│
├── steps/                            # Pipeline steps
│   ├── select_input.py               # Source specification
│   ├── fetch_source.py               # Repository fetching  
│   ├── analyze_code.py               # Code analysis & selection
│   ├── gen_tests_agent.py            # LLM test generation
│   ├── gen_tests_baseline.py         # Heuristic test generation
│   ├── run_tests.py                  # Test execution & coverage
│   ├── evaluate_coverage.py          # Metrics & gate evaluation
│   ├── compare_and_promote.py        # Model registry promotion
│   ├── resolve_test_pack.py          # Test pack resolution
│   └── report.py                     # Report generation
│
├── prompts/                          # Jinja2 prompt templates
│   ├── unit_test_v1.jinja           # Standard test generation
│   └── unit_test_strict_v2.jinja    # Comprehensive test generation
│
├── materializers/                    # Custom artifact handling
├── utils/                           # Utility functions
│
├── registry/                        # Test Pack registry docs
│   └── README.md
│
├── run_templates/                   # Experiment templates
│   ├── ab_agent_vs_strict.json    # A/B testing configuration
│   └── baseline_only.json         # Baseline establishment
│
├── scripts/                        # CLI scripts
│   ├── run_experiment.py          # Experiment runner
│   └── run_batch.py              # Batch regression runner
│
└── examples/                       # Demo code for testing
    └── toy_lib/                   # Sample library
        ├── calculator.py
        └── string_utils.py
```

### Key Components

- **Domain Models**: Pydantic schemas for type safety and validation
- **Pipeline Steps**: Modular, reusable components with clear interfaces
- **Prompt Templates**: Jinja2 templates for LLM test generation  
- **Configuration**: YAML-driven experiment and deployment settings
- **Quality Gates**: Configurable thresholds for coverage and promotion
- **Model Registry**: ZenML Model Registry integration for test pack versioning

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

Set up automated regression testing:

```bash
# Register schedule (example with ZenML Cloud)
python scripts/run_batch.py --config configs/schedule.batch.yaml --schedule

# Monitor via dashboard:
# - Daily regression results
# - Coverage trend analysis  
# - Test pack performance
```

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
python scripts/run_experiment.py --config configs/experiment.default.yaml
```

## 📚 Resources

- [ZenML Documentation](https://docs.zenml.io/)
- [Model Control Plane](https://docs.zenml.io/user-guide/model-control-plane)
- [Kubernetes Orchestrator](https://docs.zenml.io/stacks/stack-components/orchestrators/kubernetes)

---

Built with ❤️ using [ZenML](https://zenml.io) - *The MLOps Framework for Production AI*