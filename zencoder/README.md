<div align="center">

  <!-- PROJECT LOGO -->
  <br />
    <a href="https://zenml.io">
      <img alt="ZenCoder Header" src=".assets/zencoder_header.png" alt="ZenML Logo">
    </a>
  <br />

</div>

<div align="center">
  <h3 align="center">ZenCoder: MLOps pipelines to train and deploy a model to produce MLOps pipelines.</h3>
  <p align="center">
    <div align="center">
      Join our <a href="https://zenml.io/slack-invite" target="_blank">
      <img width="18" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
    <b>Slack Community</b> </a> and join the zencoder channel!
    </div>
    <br />
  </p>
</div>

---

# ☮️ Automate MLOps pipeline development with ZenCoder

One of the first jobs of somebody entering MLOps is to convert their manual scripts or notebooks into pipelines that can be deployed on the cloud. This job is tedious, and can take time. For example, one has to think about:

1. Breaking down things into [step functions](https://docs.zenml.io/user-guide/starter-guide/create-an-ml-pipeline)
2. Type annotating the steps properly
3. Connecting the steps together in a pipeline
4. Creating the appropriate YAML files to [configure your pipeline](https://docs.zenml.io/user-guide/production-guide/configure-pipeline)
5. Developing a Dockerfile or equivalent to encapsulate [the environment](https://docs.zenml.io/user-guide/advanced-guide/environment-management/containerize-your-pipeline).

Frameworks like [ZenML](https://github.com/zenml-io/zenml) go a long way in alleviating this burden by abstracting much of the complexity away. However, recent advancement in Large Language Model based Copilots offer hope that even more repetitive aspects of this task can be automated.

Unfortuantely, most open source or proprietary models like GitHub Copilot are often lagging behind the most recent versions of ML libraries, therefore giving errorneous our outdated syntax when asked simple commands.

The goal of this project is fine-tune an open-source LLM that performs better than off-the-shelf solutions on giving the right output for the latest version of ZenML.

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
python run.py --feature-engineering --config <NAME_OF_CONFIG_IN_CONFIGS_FOLDER>
python run.py --feature-engineering --config generate_code_dataset.yaml

# Training
python run.py --training-pipeline --config <NAME_OF_CONFIG_IN_CONFIGS_FOLDER>
python run.py --training-pipeline --config finetune_gcp.yaml

# Deployment
python run.py --deployment-pipeline --config <NAME_OF_CONFIG_IN_CONFIGS_FOLDER>
python run.py --deployment-pipeline --config finetune_gcp.yaml
```

The `feature_engineering` and `deployment` pipeline can be run simply with the `default` stack, but the training pipelines [stack](https://docs.zenml.io/user-guide/production-guide/understand-stacks) will depend on the config.

The `deployment` pipelines relies on the `training_pipeline` to have run before.

## 🥇Recent developments

A working prototype has been trained and deployed as of Jan 19 2024. The model is using minimal data and finetuned using QLoRA and PEFT. The model was trained using 1 A100 GPU on the cloud:

- Training dataset [Link](https://huggingface.co/datasets/htahir1/zenml-codegen-v1)
- PEFT Model [Link](https://huggingface.co/htahir1/peft-lora-zencoder15B-personal-copilot/)
- Fully merged model (Ready to deploy on HuggingFace Inference Endpoints) [Link](https://huggingface.co/htahir1/peft-lora-zencoder15B-personal-copilot-merged)

The Weights & Biases logs for the latest training runs are available here: [Link](https://wandb.ai/zenmlcode/zenml-projects-zencoder?workspace=user-zenmlcodemonkey)

The [ZenML Cloud](https://zenml.io/cloud) was used to manage the pipelines, models, and deployments. Here are some screenshots of the process:

<div align="center">
    <img src=".assets/zencoder_mcp_1.png">
</div>

<div align="center">
    <img src=".assets/zencoder_mcp_2.png">
</div>

## 📓 To Do

This project recently did a [call of volunteers](https://www.linkedin.com/feed/update/urn:li:activity:7150388250178662400/). This TODO list can serve as a source of collaboration. If you want to work on any of the following, please [create an issue on this repository](https://github.com/zenml-io/zenml-projects/issues) and assign it to yourself!

- [x] Create a functioning data generation pipeline (initial dataset with the core [ZenML repo](https://github.com/zenml-io/zenml) scraped and pushed [here](https://huggingface.co/datasets/htahir1/zenml-codegen-v1))
- [x] Deploy the model on a [HuggingFace inference endpoint](https://ui.endpoints.huggingface.co/welcome) and use it in the [VS Code Extension](https://github.com/huggingface/llm-vscode#installation) using a deployment pipeline.
- [x] Create a functioning training pipeline.
- [ ] Curate a set of 5-10 repositories that are using the ZenML latest syntax and use data generation pipeline to push dataset to HuggingFace.
- [ ] Create a Dockerfile for the training pipeline with all requirements installed including ZenML, torch, CUDA etc. CUrrently I am having trouble creating this in this [config file](configs/finetune.yaml). Probably might make sense to create a docker imag with the right CUDA and requirements including ZenML. See here: https://sdkdocs.zenml.io/0.54.0/integration_code_docs/integrations-aws/#zenml.integrations.aws.flavors.sagemaker_step_operator_flavor.SagemakerStepOperatorSettings
- [ ] Tests trained model on various metrics
- [ ] Create a custom [model deployer](https://docs.zenml.io/stacks-and-components/component-guide/model-deployers) that deploys a huggingface model from the hub to a huggingface inference endpoint. This would involve creating a [custom model deployer](https://docs.zenml.io/stacks-and-components/component-guide/model-deployers/custom) and editing the [deployment pipeline accordingly](pipelines/deployment.py)

## :bulb: More Applications

While the work here is solely based on the task of finetuning the model for the ZenML library, the pipeline can be changed with minimal effort to point to any set of repositories on GitHub. Theoretically, one could extend this work to point to proprietary codebases to learn from them for any use-case.

For example, see how [VMWare fine-tuned StarCoder to learn their style](https://octo.vmware.com/fine-tuning-starcoder-to-learn-vmwares-coding-style/). 

Also, make sure to join our <a href="https://zenml.io/slack" target="_blank">
    <img width="15" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
    <b>Slack Community</b> 
</a> to become part of the ZenML family!