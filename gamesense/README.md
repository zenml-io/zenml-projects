# 🎮 GameSense: An LLM That Transforms Gaming Conversations into Structured Data

GameSense is a specialized language model that converts unstructured gaming conversations into structured, actionable data. It listens to how gamers talk and extracts valuable information that can power recommendations, support systems, and analytics.

## 🎯 What GameSense Does

**Input**: Gamers' natural language about games from forums, chats, reviews, etc.

**Output**: Structured data with categorized information about games, platforms, preferences, etc.

Here's a concrete example from our training data:

### Input Example (Gaming Conversation)
```
"Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac."
```

### Output Example (Structured Information)
```
inform(
    name[Dirt: Showdown],
    release_year[2012],
    esrb[E 10+ (for Everyone 10 and Older)],
    genres[driving/racing, sport],
    platforms[PlayStation, Xbox, PC],
    available_on_steam[no],
    has_linux_release[no],
    has_mac_release[no]
)
```

This structured output can be used to:
- Answer specific questions about games ("Is Dirt: Showdown available on Mac?")
- Track trends in gaming discussions
- Power recommendation engines
- Extract user opinions and sentiment
- Build gaming knowledge graphs
- Enhance customer support

## 🚀 How GameSense Transforms Gaming Conversations

GameSense listens to gaming chats, forum posts, customer support tickets, social media, and other sources where gamers communicate. As gamers discuss different titles, features, opinions, and issues, GameSense:

1. **Recognizes gaming jargon** across different genres and communities
2. **Extracts key information** about games, platforms, features, and opinions
3. **Structures this information** into a standardized format
4. **Makes it available** for downstream applications

## 💡 Real-World Applications

### Community Analysis
Monitor conversations across Discord, Reddit, and other platforms to track what games are being discussed, what features players care about, and emerging trends.

### Intelligent Customer Support
When a player says: "I can't get Dirt: Showdown to run on my Mac," GameSense identifies:
- The specific game (Dirt: Showdown)
- The platform issue (Mac)
- The fact that the game doesn't support Mac (from structured knowledge)
- Can immediately inform the player about platform incompatibility

### Smart Recommendations
When a player has been discussing racing games for PlayStation with family-friendly ratings, GameSense can help power recommendations for similar titles they might enjoy.

### Automated Content Moderation
By understanding the context of gaming conversations, GameSense can better identify toxic behavior while recognizing harmless gaming slang.

## 🧠 Technical Approach

GameSense uses Parameter-Efficient Fine-Tuning (PEFT) to customize powerful foundation models for understanding gaming language:

1. We start with a base model like Microsoft's Phi-2 or Llama 3.1
2. Fine-tune on the gem/viggo dataset containing structured gaming conversations
3. Use LoRA adapters for efficient training
4. Evaluate on gaming-specific benchmarks
5. Deploy to production environments

<div align="center">
  <br/>
    <a href="https://cloud.zenml.io">
      <img alt="GameSense Pipeline" src=".assets/pipeline.png" width="70%">
    </a>
  <br/>
</div>

## 🎯 Use Cases

- **Customer Support Automation**: Understand and respond to player issues with context-aware solutions
- **Community Moderation**: Detect toxic language with nuanced understanding of gaming communication
- **Player Insights**: Extract actionable intelligence from forums, chats, and reviews
- **Recommendation Systems**: Power personalized game and content suggestions
- **In-Game Assistants**: Create NPCs or helpers that understand player intentions

## 🔧 Getting Started

### Prerequisites

- Python 3.8+
- GPU with at least 24GB VRAM (for full model training)
- ZenML installed and configured
- Neptune.ai account for experiment tracking (optional)

### Environment Setup

1. Set up your Neptune.ai credentials if you want to use Neptune for experiment tracking:
   ```bash
   # Set your Neptune project name and API token as environment variables
   export NEPTUNE_PROJECT="your-neptune-workspace/your-project-name"
   export NEPTUNE_API_TOKEN="your-neptune-api-token"
   ```

### Quick Setup

1. Install GameSense:
   ```bash
   # Set up a Python virtual environment
   python3 -m venv .venv
   source .venv/bin/activate

   # Install requirements
   pip install -r requirements.txt
   ```

2. Run the end-to-end pipeline:
   ```shell
   # For single-GPU training
   python run.py --config orchestrator_finetune.yaml

   # For multi-GPU acceleration
   python run.py --config orchestrator_finetune.yaml --accelerate
   ```

> [!WARNING]  
> All pipeline steps have a `clean_gpu_memory(force=True)` at the beginning. This ensures memory is properly cleared after previous steps.
> 
> This functionality might affect other GPU processes running on the same environment. If you don't want to clean GPU memory between steps, you can remove these utility calls from all steps.

The trained model will be automatically stored in your ZenML artifact store, ready for deployment.

## ⚙️ Configuration Options

GameSense offers flexible configuration to meet your specific gaming platform needs:

### Base Model Selection

Choose from powerful foundation models:
- **Microsoft Phi-2**: Lightweight yet powerful for most gaming applications (default)
- **Llama 3.1**: Advanced capabilities for complex gaming interactions

```shell
# To use Llama 3.1 instead of Phi-2
python run.py --config configs/llama3-1_finetune_local.yaml
```

> [!TIP]  
> To finetune the Llama 3.1 base model, use the alternative configuration files provided in the `configs` folder:
> - For remote finetuning: [`llama3-1_finetune_remote.yaml`](configs/llama3-1_finetune_remote.yaml)
> - For local finetuning: [`llama3-1_finetune_local.yaml`](configs/llama3-1_finetune_local.yaml)

### Dataset Configuration

By default, GameSense uses the gem/viggo dataset, which contains structured gaming information like:

| gem_id | meaning_representation | target | references |
|--------|------------------------|--------|------------|
| viggo-train-0 | inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no]) | Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac. | [Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.] |
| viggo-train-1 | inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+...]) | Dirt: Showdown is a sport racing game... | [Dirt: Showdown is a sport racing game...] |

You can also train on your own gaming conversations by formatting them in a similar structure and updating the configuration.

### Training Acceleration

For faster training on high-end hardware:
- **Multi-GPU Training**: Distribute training across multiple GPUs using Distributed Data Parallelism (DDP)
- **Mixed Precision**: Optimize memory usage without sacrificing quality

```shell
# Enable distributed training across all available GPUs
python run.py --config orchestrator_finetune.yaml --accelerate
```

Under the hood, the finetuning step will spin up an accelerated job using Hugging Face Accelerate, which will run on all available GPUs.

## ☁️ Enterprise Deployment

For production deployment, GameSense can be trained and served on cloud infrastructure:

1. **Set up your cloud environment**:
   - Register an [orchestrator](https://docs.zenml.io/stack-components/orchestrators) or [step operator](https://docs.zenml.io/stack-components/step-operators) with GPU access (at least 24GB VRAM)
   - Register a remote [artifact store](https://docs.zenml.io/stack-components/artifact-stores) and [container registry](https://docs.zenml.io/stack-components/container-registries)
   - To access GPUs with sufficient VRAM, you may need to increase your GPU quota ([AWS](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html), [GCP](https://console.cloud.google.com/iam-admin/quotas), [Azure](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2#request-quota-and-limit-increases))
   - If the CUDA version on your GPU instance is incompatible with the default Docker image, modify it in the configuration file. See [available PyTorch images](https://hub.docker.com/r/pytorch/pytorch/tags)

   ```shell
   # Register a complete stack with GPU support
   zenml stack register gamesense-stack -o <ORCHESTRATOR_NAME> \
       -a <ARTIFACT_STORE_NAME> \
       -c <CONTAINER_REGISTRY_NAME> \
       [-s <STEP_OPERATOR_NAME>]
   ```

2. **Launch remote training**:
   ```shell
   # For cloud-based training
   python run.py --config configs/llama3-1_finetune_remote.yaml
   ```

## 🔄 Customization for Your Gaming Platform

### Training on Your Gaming Data

To fine-tune GameSense on your specific gaming platform's data:

1. **Prepare your dataset**: Format your gaming conversations, support tickets, or forum posts
2. **Update the configuration**: Modify the `dataset_name` parameter in your config file
3. **Adjust tokenization**: If needed, customize the [`generate_and_tokenize_prompt`](utils/tokenizer.py) function

For detailed instructions on data preparation, see our [data customization guide](#️-bring-your-own-gaming-data).

## 📊 Performance Monitoring

GameSense includes built-in evaluation using industry-standard metrics:

- **ROUGE Scores**: Measure how well the model can generate natural language from structured data
- **Gaming-Specific Benchmarks**: Evaluate understanding of gaming terminology
- **Automatic Model Promotion**: Only deploy models that meet quality thresholds

All metrics are tracked in the ZenML dashboard for easy monitoring and comparison.

<div align="center">
  <br/>
    <a href="https://cloud.zenml.io">
      <img alt="Model Control Plane" src=".assets/model.png">
    </a>
  <br/>
</div>

## 📁 Technical Architecture

GameSense follows a modular architecture for easy customization:

```
├── configs                                       # Configuration profiles for different deployment scenarios
│   ├── orchestrator_finetune.yaml                # Default local or remote orchestrator configuration
│   └── remote_finetune.yaml                      # Default step operator configuration
├── materializers                                 # Custom data handlers for gaming-specific content
│   └── directory_materializer.py                 # Custom materializer to push directories to the artifact store
├── pipelines                                     # Core pipeline definitions
│   └── train.py                                  # Finetuning and evaluation pipeline
├── steps                                         # Individual pipeline components
│   ├── evaluate_model.py                         # Gaming-specific evaluation metrics
│   ├── finetune.py                               # Model customization for gaming terminology
│   ├── log_metadata.py                           # Helper step for model metadata logging
│   ├── prepare_datasets.py                       # Gaming data processing
│   └── promote.py                                # Production deployment logic
├── utils                                         # Utility functions
│   ├── callbacks.py                              # Custom callbacks
│   ├── loaders.py                                # Loaders for models and data
│   ├── logging.py                                # Logging helpers
│   └── tokenizer.py                              # Load and tokenize
└── run.py                                        # CLI tool to run pipelines on ZenML Stack
```

## 🗂️ Bring Your Own Gaming Data

To fine-tune GameSense on your specific gaming platform's data:

1. **Format your dataset**: Prepare your gaming conversations in a structured format similar to gem/viggo
2. **Update the configuration**: Point to your dataset in the config file
3. **Run the pipeline**: GameSense will automatically process and learn from your data

The [`prepare_data` step](steps/prepare_datasets.py) handles:
- Loading, tokenizing, and storing the dataset from an external source to your artifact store
- Loading datasets from Hugging Face (requires `train`, `validation`, and `test` splits by default)
- Tokenization via the [`generate_and_tokenize_prompt`](utils/tokenizer.py) utility function

For custom data sources, you'll need to prepare the splits in a Hugging Face dataset format. The step returns paths to the stored datasets (`train`, `val`, and `test_raw` splits), with the test set tokenized later during evaluation.

You can structure conversations from:
- Game forums
- Support tickets
- Discord chats
- Streaming chats
- Reviews
- Social media posts

## 📚 Documentation

For learning more about how to use ZenML to build your own MLOps pipelines, refer to our comprehensive [ZenML documentation](https://docs.zenml.io/).

## Running on CPU-only Environment

If you don't have access to a GPU, you can still run this project with the CPU-only configuration. We've made several optimizations to make this project work on CPU, including:

- Smaller batch sizes for reduced memory footprint
- Fewer training steps
- Disabled GPU-specific features (quantization, bf16, etc.)
- Using smaller test datasets for evaluation
- Special handling for Phi-3.5 model caching issues on CPU

To run the project on CPU:

```bash
python run.py --config phi3.5_finetune_cpu.yaml
```

Note that training on CPU will be significantly slower than training on a GPU. The CPU configuration uses:

1. A smaller model (`phi-3.5-mini-instruct`) which is more CPU-friendly
2. Reduced batch size and increased gradient accumulation steps
3. Fewer total training steps (50 instead of 300)
4. Half-precision (float16) where possible to reduce memory usage
5. Smaller dataset subsets (100 training samples, 20 validation samples, 10 test samples)
6. Special compatibility settings for Phi models running on CPU

For best results, we recommend:
- Using a machine with at least 16GB of RAM
- Being patient! LLM training on CPU is much slower than on GPU
- If you still encounter memory issues, try reducing the `max_train_samples` parameter even further in the config file

### Known Issues and Workarounds

Some large language models like Phi-3.5 have caching mechanisms that are optimized for GPU usage and may encounter issues when running on CPU. Our CPU configuration includes several workarounds:

1. Disabling KV caching for model generation
2. Using `torch.float16 data` type to reduce memory usage
3. Disabling flash attention which isn't needed on CPU
4. Using standard AdamW optimizer instead of 8-bit optimizers that require GPU

These changes allow the model to run on CPU with less memory and avoid compatibility issues, although at the cost of some performance.
