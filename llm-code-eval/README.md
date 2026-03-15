# LLM Code Evaluation Pipeline

A ZenML pipeline that evaluates coding LLMs on HumanEval problems, comparing
multiple models side-by-side with LLM-as-judge scoring.

## What It Does

1. **Loads test cases** — curated subset of HumanEval (15-20 Python problems
   across easy/medium/hard difficulty)
2. **Runs inference** — fan-out across all (test_case, model) pairs using
   ZenML's `.product()` dynamic pipeline feature
3. **Evaluates outputs** — LLM-as-judge scores each completion on correctness,
   style, and completeness (1-5 scale)
4. **Generates report** — HTML comparison matrix rendered in the ZenML dashboard

## Key ZenML Features Showcased

- **Dynamic pipelines** with `.product()` for cartesian fan-out/fan-in
- **LiteLLM integration** — swap models via YAML config
- **Step caching** — re-run evaluations without re-inferring
- **HTML artifact visualization** — comparison report in the dashboard
- **Metadata logging** — per-model metrics visible in the ZenML UI

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default config (2 models, 15 problems)
python run.py

# Quick test with just DeepSeek (5 problems)
python run.py --config configs/deepseek_only.yaml

# Full comparison (4 models, 16 problems)
python run.py --config configs/full_comparison.yaml

# Disable caching for a fresh run
python run.py --no-cache
```

## Configuration

Configs live in `configs/` and control:

| Config | Models | Problems | Use Case |
|--------|--------|----------|----------|
| `default.yaml` | 2 | 15 | Development / quick runs |
| `deepseek_only.yaml` | 1 | 5 | Smoke test |
| `full_comparison.yaml` | 4 | 16 | Full benchmark |

### Environment Variables

Set API keys for the models you want to evaluate:

```bash
export DEEPSEEK_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."       # for full_comparison
export GEMINI_API_KEY="..."       # for full_comparison

# Optional: Langfuse tracing
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="..."
```

## Project Structure

```
llm-code-eval/
├── run.py                      # CLI entrypoint
├── configs/                    # YAML pipeline configs
├── pipelines/
│   └── code_eval_pipeline.py   # Pipeline definition with .product()
├── steps/
│   ├── load_test_cases.py      # Load curated HumanEval subset
│   ├── run_inference.py        # LiteLLM inference with metrics
│   ├── evaluate_outputs.py     # LLM-as-judge scoring
│   └── generate_report.py      # HTML report generation
├── materializers/
│   └── html_report_materializer.py  # Dashboard visualization
├── utils/
│   ├── scoring.py              # Pydantic data models
│   ├── litellm_utils.py        # LiteLLM wrapper with cost tracking
│   └── html_templates.py       # Report HTML generation
├── test_cases/                 # Curated HumanEval problems (JSON)
└── scripts/
    └── regenerate_humaneval_subset.py  # Refresh from HuggingFace
```

## Adapting for Your Use Case

- **Add models**: Edit the `models` list in any config YAML
- **Change problems**: Edit JSON files in `test_cases/` or regenerate with
  `uv run scripts/regenerate_humaneval_subset.py`
- **Adjust scoring**: Modify the judge prompt in `steps/evaluate_outputs.py`
- **Custom metrics**: Extend `JudgeScore` in `utils/scoring.py`
