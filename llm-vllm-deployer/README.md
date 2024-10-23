# ☮️ Deploying open source LLMs using MLOps pipelines with vLLM

Welcome to your newly generated "ZenML LLM vLLM deployment project" project! This is
a great way to get hands-on with ZenML using production-like template.
The project contains a collection of ZenML steps, pipelines and other artifacts
and useful resources that can serve as a solid starting point for deploying open-source LLMs using ZenML.

Using these pipelines, we can run the data-preparation and model finetuning with a single command while using YAML files for [configuration](https://docs.zenml.io/user-guide/production-guide/configure-pipeline) and letting ZenML take care of tracking our metadata and [containerizing our pipelines](https://docs.zenml.io/how-to/customize-docker-builds).

<TODO: Add image from ZenML Cloud for pipeline here>

## 🏃 How to run

In this project, we will deploy the [gpt-2](https://huggingface.co/openai-community/gpt2) model using [vLLM](https://github.com/vllm-project/vllm). Before we're able to run any pipeline, we need to set up our environment as follows:

```bash
# Set up a Python virtual environment, if you haven't already
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

Run the deployment pipeline

```bash
python run.py
```

## 📜 Project Structure

The project loosely follows [the recommended ZenML project structure](https://docs.zenml.io/how-to/setting-up-a-project-repository/best-practices):

```
.
├── configs                                      # pipeline configuration files
│   ├── default_vllm_deploy.yaml                 # default local or remote orchestrator configuration
├── pipelines                                    # `zenml.pipeline` implementations
│   └── deploy_pipeline.py                       # vllm deployment pipeline
├── steps                                        # logically grouped `zenml.steps` implementations
│   ├── vllm_deployer.py                         # deploy model using vllm
├── README.md                                    # this file
├── requirements.txt                             # extra Python dependencies 
└── run.py                                       # CLI tool to run pipelines on ZenML Stack
```
