# README for LLM Fine-Tuning with ZenML and Lightning AI Studios

## Overview

In the fast-paced world of AI, the ability to efficiently fine-tune Large Language Models (LLMs) for specific tasks is crucial. This project combines ZenML with Lightning AI Studios to streamline and automate the LLM fine-tuning process, enabling rapid iteration and deployment of task-specific models. This is a toy showcase only but can be extended for full production use.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Running the Pipeline](#running-the-pipeline)
4. [Configuration](#configuration)
5. [Accelerated Fine-Tuning](#accelerated-fine-tuning)
6. [Running with Remote Stack](#running-with-remote-stack)
7. [Customizing Data Preparation](#customizing-data-preparation)
8. [Project Structure](#project-structure)
9. [Benefits & Future](#benefits--future)
10. [Credits](#credits)

## Introduction

As LLMs such as GPT-4, Llama 3.1, and Mistral become more accessible, companies aim to adapt these models for specialized tasks like customer service chatbots, content generation, and specialized data analysis. This project addresses the challenge of scaling fine-tuning and managing numerous LLM variants by combining Lightning AI Studios with the automation capabilities of ZenML.

### Key Benefits

- **Efficient Fine-Tuning:** Fine-tune models with minimal computational resources.
- **Ease of Management:** Store and distribute adapter weights efficiently.
- **Scalability:** Serve thousands of fine-tuned variants from a single base model.

## Installation

To set up your environment, follow these steps:

```bash
# Set up a Python virtual environment, if you haven't already
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install ZenML and Lightning integrations
pip install zenml
zenml integration install lightning s3 aws -y

# Initialize and connect to a deployed ZenML server
zenml init
zenml login <MYZENMLSERVERURL>
```

## Running the Pipeline

To run the fine-tuning pipeline, use the `run.py` script with the appropriate configuration file:

```shell
python run.py --config configs/config_large_gpu.yaml
```

## Configuration

The fine-tuning process can be configured using YAML files located in the `configs` directory. Here are examples:

### Example `config_large_gpu.yaml`

```yaml
model:
  name: llm-finetuning-gpt2-large
  description: "Fine-tune GPT-2 on larger GPU."
  tags:
    - llm
    - finetuning
    - gpt2-large

parameters:
  base_model_id: gpt2-large

steps:
  prepare_data:
    parameters:
      dataset_name: squad
      dataset_size: 1000
      max_length: 512

  finetune:
    parameters:
      num_train_epochs: 3
      per_device_train_batch_size: 8

    settings:
      orchestrator.lightning:
        machine_type: A10G
```

### Example `config_small_cpu.yaml`

```yaml
model:
  name: llm-finetuning-distilgpt2-small
  description: "Fine-tune DistilGPT-2 on smaller computer."
  tags:
    - llm
    - finetuning
    - distilgpt2

parameters:
  base_model_id: distilgpt2

steps:
  prepare_data:
    parameters:
      dataset_name: squad
      dataset_size: 100
      max_length: 128

  finetune:
    parameters:
      num_train_epochs: 1
      per_device_train_batch_size: 4
```

## Running with Remote Stack

Set up a remote lightning stack with ZenML for fine tuning on remote infrastructure:

1. **Register Orchestrator and Artifact Store:**

    ```shell
    zenml integration install lightning s3
    zenml orchestrator register lightning_orchestrator --flavor=lightning --machine_type=CPU --user_id=<YOUR_LIGHTNING_USER_ID> --api_key=<YOUR_LIGHTNING_API_KEY> --username=<YOUR_LIGHTNING_USERNAME>
    zenml artifact-store register s3_store --flavor=s3 --path=s3://yourpath
    ```

2. **Set up and Register the Stack:**

    ```shell
    zenml stack register lightning_stack -o lightning_orchestrator -a s3_store
    zenml stack set lightning_stack
    ```

## Customizing Data Preparation

Customize the `prepare_data` step for different datasets by modifying loading logic or tokenization patterns. Update the relevant YAML configuration parameters to fit your dataset and requirements.

## Project Structure

The project follows a structured layout for easy navigation and management:

```
.
├── configs                                          # Configuration files for the pipeline
│   ├── config_large_gpu.yaml                        # Config for large GPU setup
│   ├── config_small_cpu.yaml                        # Config for small CPU setup
├── .dockerignore                                    # Docker ignore file
├── LICENSE                                          # License file
├── README.md                                        # This file
├── requirements.txt                                 # Python dependencies
├── run.py                                           # CLI tool to run pipelines on ZenML Stack
```

## Benefits & Future

Using smaller, task-specific models is more efficient and cost-effective than relying on large general-purpose models. This strategy allows for:

- **Cost-Effectiveness:** Less computational resources reduce operational costs.
- **Improved Performance:** Models fine-tuned on specific data often outperform general models on specialized tasks.
- **Faster Iteration:** Quicker experimentation and iteration cycles.
- **Data Privacy:** Control over training data, crucial for industries with strict privacy requirements.

## Credits

This project relies on several tools and libraries:

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [ZenML](https://zenml.io/)
- [Lightning AI Studios](https://www.lightning.ai/)

With these tools, you can efficiently manage the lifecycle of multiple fine-tuned LLM variants, benefiting from the robust infrastructure provided by ZenML and the scalable resources of Lightning AI Studios.

For more details, consult the [ZenML documentation](https://docs.zenml.io) and the [Lightning AI Studio documentation](https://lightning.ai).