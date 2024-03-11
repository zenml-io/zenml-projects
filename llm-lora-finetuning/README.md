# ☮️ Fine-tuning open source LLMs using MLOps pipelines

The goal of this project is to use [ZenML](https://github.com/zenml-io/zenml) to write reusable MLOps pipelines to fine-tune various opens source LLMs.

Using these pipelines, we can run the data-preparation and model finetuning with a single command while using YAML files for [configuration](https://docs.zenml.io/user-guide/production-guide/configure-pipeline) and letting ZenML take care of tracking our metadata and [containerizing our pipelines](https://docs.zenml.io/user-guide/advanced-guide/infrastructure-management/containerize-your-pipeline).

## :earth_americas: Inspiration and Credit

This project heavily relies on the [Lit-GPT project](https://github.com/Lightning-AI/litgpt) of the amazing people at Lightning AI. We used [this blogpost](https://lightning.ai/pages/community/lora-insights/#toc14) to get started with LoRA and QLoRA and modified the commands they recommend to make them work using ZenML.

## 🏃 How to run

In this repository we provide a few predefined configuration files for finetuning the [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset. You can change both the base model and dataset by modifying the configuration files.

If you want to push any of your finetuned adapters or merged models to huggingface, you will need to register a secret with your huggingface access token as follows:
```shell
zenml secret create huggingface_credentials --token=<HUGGINGFACE_TOKEN>
```

### Combined feature engineering and finetuning pipeline

The easiest way to get started with just a single command is to run the finetuning pipeline with the `finetune-mistral-alpaca.yaml` configuration file, which will do both feature engineering and finetuning:

```shell
python run.py --finetuning-pipeline --config finetune-mistral-alpaca.yaml
```

When running the pipeline like this, the trained adapter will be stored in the ZenML artifact store. You can optionally upload the adapter, the merged model or both by specifying the `adapter_output_repo` and `merged_output_repo` parameters in the configuration file.


### Evaluation pipeline

Before running this pipeline, you will need to fill in the `adapter_repo` in the `eval-mistral.yaml` configuration file. This should point to a huggingface repository that contains the finetuned adapter you got by running the finetuning pipeline.

```shell
python run.py --eval-pipeline --config eval-mistral.yaml
```

### Merging pipeline

In case you have trained an adapter using the finetuning pipeline, you can merge it with the base model by filling in the `adapter_repo` and `output_repo` parameters in the `merge-mistral.yaml` file, and then running:

```shell
python run.py --merge-pipeline --config merge-mistral.yaml
```

### Feature Engineering followed by Finetuning

If you want to finetune your model on a different dataset, you can do so by running the feature engineering pipeline followed by the finetuning pipeline. To define your dataset, take a look at the `scripts/prepare_*` scripts and set the dataset name in the `feature-mistral-alpaca.yaml` config file.

```shell
python run.py --feature-pipeline --config --feature-mistral-alpaca.yaml
python run.py --finetuning-pipeline --config finetune-mistral-from-dataset.yaml
```
