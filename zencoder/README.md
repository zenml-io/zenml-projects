<div align="center">

  <!-- PROJECT LOGO -->
  <br />
    <a href="https://zenml.io">
      <img alt="ZenCoder Header" src=".assets/zencoder_header.png" alt="ZenML Logo">
    </a>
  <br />

</div>

<div align="center">
  <h3 align="center">ZenCoder: LLMOps pipelines to train and deploy a model to produce MLOps pipelines.</h3>
  <p align="center">
    Transform your ML workflow with an AI assistant that actually understands ZenML. This project fine-tunes open-source LLMs to generate production-ready MLOps pipelines.
    <div align="center">
      Join our <a href="https://zenml.io/slack-invite" target="_blank">
      <img width="18" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
    <b>Slack Community</b> </a> and join the zencoder channel!
    </div>
    <br />
  </p>
</div>

---

# ☮️ ZenCoder: Fine-tuning an open source LLM to create MLOps pipelines

One of the first jobs of somebody entering MLOps is to convert their manual scripts or notebooks into pipelines that can be deployed on the cloud. This job is tedious, and can take time. For example, one has to think about:

1. Breaking down things into [step functions](https://docs.zenml.io/user-guides/starter-guide/create-an-ml-pipeline)
2. Type annotating the steps properly
3. Connecting the steps together in a pipeline
4. Creating the appropriate YAML files to [configure your pipeline](https://docs.zenml.io/user-guides/production-guide/configure-pipeline)
5. Developing a Dockerfile or equivalent to encapsulate [the environment](https://docs.zenml.io/how-to/customize-docker-builds).

Frameworks like [ZenML](https://github.com/zenml-io/zenml) go a long way in alleviating this burden by abstracting much of the complexity away. However, recent advancement in Large Language Model based Copilots offer hope that even more repetitive aspects of this task can be automated.

Unfortunately, most open source or proprietary models like GitHub Copilot are often lagging behind the most recent versions of ML libraries, therefore giving erroneous our outdated syntax when asked simple commands.

ZenCoder is designed to solve this problem by fine-tuning an open-source LLM that performs better than off-the-shelf solutions on giving the right output for the latest version of ZenML. Think of it as your personal MLOps engineer that understands ZenML deeply and can help you build production-ready pipelines.

## :earth_americas: Inspiration and Credit

For this purpose of this project, we are going to be leveraging the excellent work of [Sourab Mangrulkar](https://huggingface.co/smangrul) and [Sayak Paul](https://huggingface.co/sayakpaul), who fine-tuned the [StarCoder](https://huggingface.co/bigcode/starcoder) model on the latest version of HuggingFace. They summarized their work in [this blog post on HuggingFace](https://huggingface.co/blog/personal-copilot).

Our [data generation pipeline](pipelines/generate_code_dataset.py) is based on the [codegen](https://github.com/sayakpaul/hf-codegen) repository, and the [training pipeline](pipelines/) is based on [this script](https://github.com/pacman100/DHS-LLM-Workshop/blob/main/personal_copilot/training/train.py). All credit to Sourab and Sayak for putting this work together!

## 🧑‍✈️ Train your own copilot

The work presented in this repository can easily be extended to other codebases and use-cases than just helping ML Engineering. You can easily modify the pipelines to point to other private codebases, and train a personal copilot on your codebase!

See the [data generation pipeline](pipelines/generate_code_dataset.py) as a starting point.

## 🍍Methodology

Now, we could take the code above and run it as scripts on some chosen ZenML repositories. But just to make it a bit more fun, we're going to be building ZenML pipelines to achieve this task!

That way we write ZenML pipelines to train a model that can produce ZenML pipelines 🐍. Sounds fun.

Specifically, we aim to create three pipelines:

- The data generation pipeline ([here](pipelines/generate_code_dataset.py)) that scrapes a chosen set of latest zenml version based repositories on GitHub, and pushes the dataset to HuggingFace.
- The training pipeline ([here](pipelines/finetune.py)) that loads the dataset from the previous pipeline and launches a training job on a cloud provider to train the model.
- The deployment pipeline ([here](pipelines/deployment.py) that deploys the model to huggingface inference endpoints)

## 🏃 How to run

The three pipelines can be run using the CLI:

```shell
# Data generation
python run.py --feature-pipeline --config <NAME_OF_CONFIG_IN_CONFIGS_FOLDER>
python run.py --feature-pipeline --config generate_code_dataset.yaml

# Training
python run.py --training-pipeline --config <NAME_OF_CONFIG_IN_CONFIGS_FOLDER>
python run.py --training-pipeline --config finetune_gcp.yaml

# Deployment
python run.py --deploy-pipeline --config <NAME_OF_CONFIG_IN_CONFIGS_FOLDER>
python run.py --deploy-pipeline --config deployment_a100.yaml
```

The `feature_engineering` and `deployment` pipeline can be run simply with the `default` stack, but the training pipelines [stack](https://docs.zenml.io/user-guides/production-guide/understand-stacks) will depend on the config.

The `deployment` pipelines relies on the `training_pipeline` to have run before.

## :cloud: Deployment

We have create a custom zenml model deployer for deploying models on the
huggingface inference endpoint. The code for custom deployer is in
the deployment pipeline which can be found [here](./pipelines/deployment.py).

The deployment pipeline supports two deployment targets:
1. **HuggingFace Inference Endpoints** - Deploy the model to HuggingFace's managed inference service
2. **Local vLLM Deployment** - Deploy the model locally using vLLM for high-performance inference

### HuggingFace Deployment

For running deployment pipeline with HuggingFace, we create a custom zenml stack. As we are using a custom model deployer, we will have to register the flavor and model deployer. We update the stack to use this custom model deployer for running deployment pipeline.

```bash
zenml init
zenml stack register zencoder_hf_stack -o default -a default
zenml stack set zencoder_hf_stack
export HUGGINGFACE_USERNAME=<here>
export HUGGINGFACE_TOKEN=<here>
export NAMESPACE=<here>
zenml secret create huggingface_creds --username=$HUGGINGFACE_USERNAME --token=$HUGGINGFACE_TOKEN
zenml model-deployer flavor register huggingface.hf_model_deployer_flavor.HuggingFaceModelDeployerFlavor
```

Afterward, you should see the new flavor in the list of available flavors:

```bash
zenml model-deployer flavor list
```

Register model deployer component into the current stack

```bash
zenml model-deployer register hfendpoint --flavor=hfendpoint --token=$HUGGINGFACE_TOKEN --namespace=$NAMESPACE
zenml stack update zencoder_hf_stack -d hfendpoint
```

Run the deployment pipeline using the CLI:

```shell
# Deployment to HuggingFace
python run.py --deployment-pipeline --config deployment_a100.yaml --deployment-target huggingface
```

### Local vLLM Deployment

For local deployment using vLLM, we leverage ZenML's vLLM integration which provides high-performance inference capabilities. This is ideal for testing the model locally before deploying to production or for scenarios where you need local inference.

First, ensure you have the vLLM integration installed:

```bash
zenml integration install vllm
```

Then register the vLLM model deployer and update your stack:

```bash
zenml model-deployer register vllm-deployer --flavor=vllm
zenml stack update zencoder_hf_stack -d vllm-deployer
```

Run the deployment pipeline with vLLM target:

```shell
# Local deployment with vLLM
python run.py --deployment-pipeline --config deployment_a100.yaml --deployment-target vllm
```

This will start a local vLLM server that serves your model, providing a REST API endpoint for inference. The server is optimized for performance and can handle multiple requests efficiently.

## 🥇Recent developments

A working prototype has been trained and deployed as of Jan 19 2024. The model is using minimal data and finetuned using QLoRA and PEFT. The model was trained using 1 A100 GPU on the cloud:

- Training dataset [Link](https://huggingface.co/datasets/zenml/zenml-codegen-v1)
- PEFT Model [Link](https://huggingface.co/htahir1/peft-lora-zencoder15B-personal-copilot/)
- Fully merged model (Ready to deploy on HuggingFace Inference Endpoints) [Link](https://huggingface.co/htahir1/peft-lora-zencoder15B-personal-copilot-merged)

The Weights & Biases logs for the latest training runs are available here: [Link](https://wandb.ai/zenmlcode/zenml-projects-zencoder?workspace=user-zenmlcodemonkey)

The [ZenML Pro](https://zenml.io/pro) was used to manage the pipelines, models, and deployments. Here are some screenshots of the process:

<div align="center">
    <img src=".assets/zencoder_mcp_1.png">
</div>

<div align="center">
    <img src=".assets/zencoder_mcp_2.png">
</div>

## 📓 To Do

This project recently did a [call of volunteers](https://www.linkedin.com/feed/update/urn:li:activity:7150388250178662400/). This TODO list can serve as a source of collaboration. If you want to work on any of the following, please [create an issue on this repository](https://github.com/zenml-io/zenml-projects/issues) and assign it to yourself!

- [x] Create a functioning data generation pipeline (initial dataset with the core [ZenML repo](https://github.com/zenml-io/zenml) scraped and pushed [here](https://huggingface.co/datasets/zenml/zenml-codegen-v1))
- [x] Deploy the model on a HuggingFace inference endpoint and use it in the [VS Code Extension](https://github.com/huggingface/llm-vscode#installation) using a deployment pipeline.
- [x] Create a functioning training pipeline.
- [ ] Curate a set of 5-10 repositories that are using the ZenML latest syntax and use data generation pipeline to push dataset to HuggingFace.
- [ ] Create a Dockerfile for the training pipeline with all requirements installed including ZenML, torch, CUDA etc. CUrrently I am having trouble creating this in this [config file](configs/finetune_local.yaml). Probably might make sense to create a docker imag with the right CUDA and requirements including ZenML. See here: https://sdkdocs.zenml.io/0.54.0/integration_code_docs/integrations-aws/#zenml.integrations.aws.flavors.sagemaker_step_operator_flavor.SagemakerStepOperatorSettings

- [ ] Tests trained model on various metrics
- [ ] Create a custom [model deployer](https://docs.zenml.io/stack-components/model-deployers) that deploys a huggingface model from the hub to a huggingface inference endpoint. This would involve creating a [custom model deployer](https://docs.zenml.io/stack-components/model-deployers/custom) and editing the [deployment pipeline accordingly](pipelines/deployment.py)

## :bulb: More Applications

While the work here is solely based on the task of finetuning the model for the ZenML library, the pipeline can be changed with minimal effort to point to any set of repositories on GitHub. Theoretically, one could extend this work to point to proprietary codebases to learn from them for any use-case.

For example, see how [VMWare fine-tuned StarCoder to learn their style](https://entreprenerdly.com/fine-tuning-starcoder-to-create-a-coding-assistant-that-adapts-to-your-coding-style/).

Also, make sure to join our <a href="https://zenml.io/slack" target="_blank">
    <img width="15" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
    <b>Slack Community</b>
</a> to become part of the ZenML family!
