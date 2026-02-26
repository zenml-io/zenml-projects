# ART Email Search Agent

Train an email search agent using [OpenPipe ART](https://github.com/openpipe/art) (Agentic Reinforcement Training) with [ZenML](https://zenml.io) for production ML pipelines.

## Overview

This project demonstrates how to:

- **Train an RL agent** using GRPO (Group Relative Policy Optimization) with RULER scoring
- **Track artifacts** including scenarios, model checkpoints, and training metrics
- **Orchestrate on Kubernetes** with GPU step operators for training
- **Evaluate models** with automated correctness judging
- **Deploy as HTTP service** using ZenML Pipeline Deployments

The agent learns to search through emails and answer questions using LangGraph's ReAct pattern, starting from a Qwen 2.5 7B base model.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ZenML Pipelines                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  data_preparation_pipeline (cached, no GPU)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ download_    │─▶│ create_      │─▶│ load_        │              │
│  │ enron_data   │  │ database     │  │ scenarios    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  training_pipeline (GPU required)                                   │
│  ┌──────────────┐  ┌────────────────────────────────┐              │
│  │ setup_art_   │─▶│ train_agent                    │              │
│  │ model        │  │ • LangGraph rollouts           │              │
│  └──────────────┘  │ • RULER scoring                │              │
│                    │ • GRPO training                │              │
│                    └────────────────────────────────┘              │
│                                                                     │
│  evaluation_pipeline (GPU required)                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ load_trained │─▶│ run_         │─▶│ compute_     │              │
│  │ _model       │  │ inference    │  │ metrics      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  inference_pipeline (HTTP Deployment)                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ POST /invoke                                                │    │
│  │   → run_single_inference → { answer, source_ids }          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for training/inference)
- OpenAI API key (for RULER scoring)

### Installation

```bash
cd art-rl
pip install -r requirements.txt  # or: uv pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```bash
# Required for RULER scoring
OPENAI_API_KEY=your-openai-key

# Optional for experiment tracking
WANDB_API_KEY=your-wandb-key
```

### Running the Pipelines

```bash
# 1. Prepare data (run once, artifacts are cached)
python run.py --pipeline data

# 2. Train the agent (requires GPU)
python run.py --pipeline train --config configs/training_local.yaml

# 3. Evaluate on test scenarios
python run.py --pipeline eval

# Or run everything:
python run.py --pipeline all
```

## Project Structure

```
art-rl/
├── run.py                    # CLI entry point
├── requirements.txt          # Dependencies
├── configs/
│   ├── data_prep.yaml        # Data preparation config
│   ├── training_local.yaml   # Local GPU training
│   ├── training_k8s.yaml     # Kubernetes training
│   ├── evaluation.yaml       # Evaluation config
│   └── deployment.yaml       # HTTP deployment config
├── pipelines/
│   ├── data_preparation.py   # Data pipeline
│   ├── training.py           # Training pipeline
│   ├── evaluation.py         # Evaluation pipeline
│   └── inference.py          # Inference pipeline (deployable)
├── steps/
│   ├── data/                 # Data preparation steps
│   ├── training/             # Training steps
│   ├── evaluation/           # Evaluation steps
│   └── inference/            # Inference steps
├── environment/
│   ├── models.py             # Pydantic data models
│   ├── email_db.py           # SQLite database operations
│   └── tools.py              # LangGraph tools
└── agent/
    ├── rollout.py            # LangGraph rollout function
    └── judge.py              # Correctness judging
```

## How It Works

### The Email Search Task

The agent is trained to answer questions about a user's email inbox. Given a question like:

> "Who can I contact for Power Operations when Sally is in London?"

The agent must:
1. Search the email database using relevant keywords
2. Read specific emails to find information
3. Return a final answer with source references

### Training with ART

[ART (Agentic Reinforcement Training)](https://art.openpipe.ai/) uses GRPO to train the agent:

1. **Rollouts**: For each training scenario, generate multiple trajectories using the LangGraph ReAct agent
2. **RULER Scoring**: An LLM judge scores trajectories relative to each other (easier than absolute scoring)
3. **GRPO Update**: Policy is updated to favor higher-scoring trajectories

### LangGraph Integration

The agent uses LangGraph's ReAct pattern with three tools:

- `search_inbox_tool`: Search emails by keywords
- `read_email_tool`: Read a specific email by ID
- `return_final_answer_tool`: Provide the final answer

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `Qwen/Qwen2.5-7B-Instruct` | Base model for fine-tuning |
| `groups_per_step` | 2 | Scenario groups per training step |
| `rollouts_per_group` | 4 | Rollouts per scenario for GRPO |
| `learning_rate` | 1e-5 | Optimizer learning rate |
| `max_steps` | 20 | Maximum training steps |
| `ruler_model` | `openai/o4-mini` | Model for RULER scoring |

### Kubernetes Training

For production training on Kubernetes with GPU nodes:

```bash
python run.py --pipeline train --config configs/training_k8s.yaml
```

The config includes:
- GPU node affinity
- Resource requests/limits
- Shared memory volume for PyTorch

### Deploying as HTTP Service

After training, deploy the agent as a real-time HTTP service:

```bash
# Deploy the inference pipeline
python run.py --pipeline deploy --name my-email-agent

# Or use ZenML CLI directly
zenml pipeline deploy pipelines.inference.inference_pipeline --name my-email-agent
```

Once deployed, invoke the service via HTTP:

```bash
curl -X POST http://localhost:8080/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "question": "What meeting is scheduled for next week?",
      "inbox_address": "john.smith@enron.com",
      "query_date": "2001-05-15"
    }
  }'
```

Response:
```json
{
  "question": "What meeting is scheduled for next week?",
  "answer": "There is a team sync scheduled for Monday at 10am.",
  "source_ids": ["<message-id-123>"],
  "success": true
}
```

Manage deployments:
```bash
zenml deployment list                    # List all deployments
zenml deployment describe my-email-agent # Show deployment details
zenml deployment logs my-email-agent -f  # Stream logs
zenml deployment deprovision my-email-agent # Stop deployment
```

## ZenML Features Used

- **Artifact Tracking**: Scenarios, checkpoints, and metrics are versioned
- **Model Control Plane**: Training metrics logged with `log_model_metadata()`
- **Docker Settings**: Custom images with CUDA and dependencies
- **Pipeline Caching**: Data preparation runs once, reused for training
- **Kubernetes Orchestration**: GPU pod settings for training steps
- **Pipeline Deployments**: Deploy inference pipelines as HTTP services

## Dataset

This project uses the [Enron Email Dataset](https://huggingface.co/datasets/corbt/enron-emails) with [sample questions](https://huggingface.co/datasets/corbt/enron_emails_sample_questions) from Hugging Face.

## References

- [ART Documentation](https://art.openpipe.ai/)
- [LangGraph Integration](https://art.openpipe.ai/integrations/langgraph-integration)
- [RULER Documentation](https://art.openpipe.ai/fundamentals/ruler)
- [ZenML Documentation](https://docs.zenml.io/)
- [ZenML Pipeline Deployments](https://docs.zenml.io/concepts/deployment)
- [Original ART Notebook](https://github.com/openpipe/art)

## License

Apache 2.0
