# Will they stay or will they go? Building a Customer Loyalty Predictor

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

**Problem statement**: For a given customer's historical data, we are asked to predict whether a customer will stop using a company's product or not. We will be using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?datasetId=13996&sortBy=voteCount) dataset for building an end to end production-grade machine learning system that can predict whether the customer will stay loyal or not. The dataset has 20 input features and a target variable for 7043 customers.

When someone leaves a company and when that customer stops paying a business for its services or products, we call that 'churn'. We can calculate a churn rate for a company by dividing the number of customers who churned by the total number of customers and then multiplying that number by 100 to reach a percentage value. If you want to learn more about customer churn, you can read this [Wikipedia article](https://en.wikipedia.org/wiki/Churn_rate).

To achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to build a production-ready pipeline that predicts whether a customer will churn or not ahead of time.
The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers your business to build and deploy machine learning pipelines in a multitude of ways:

- By offering you a framework or template to develop within.
- By integrating with popular and useful tools like Kubeflow, Seldon Core, `facets`, and more.
- By allowing you to build and deploy your machine learning pipelines easily using a modern MLOps framework.

## :snake: Python Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-churn
pip install -r requirements.txt
```

We also need to install the following ZenML integrations for this project:

```bash
zenml integration install facets sklearn xgboost lightgbm kubeflow seldon -y
```

## 📙 Resources & References

Make sure to read the blog that introduces this project in detail: [Predicting whether the customer will churn or not before they even did it](https://blog.zenml.io/customer-churn/).

# :thumbsup: The Solution

We showcase two solutions to this problem:

- `Deployment using Kubeflow Pipelines`: We will be using [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) to build and run our ZenML pipeline on the cloud and deploy it in a production environment.
- `Continuous Deployment using Seldon Core`: We will be using [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/index.html), a production-grade open-source model serving platform, to build our continuous deployment pipeline that trains a model and then serves it with Seldon Core.

## Deploy pipelines to production using orchestrator Pipelines

We will be using ZenML's [Kubeflow](https://docs.zenml.io/stack-components/orchestrators/kubeflow) integration to deploy pipelines to production using Kubeflow Pipelines on the cloud.

Our training pipeline `run_kubeflow_pipeline.py` will be built using the following steps:

- `ingest_data`: Ingest the data from the source and create a DataFrame.
- `encode_cat_cols`: Encode categorical columns.
- `drop_cols`: Dropping irrelevant columns.
- `data_splitter`: Split the data into training and test sets.
- `model_trainer`: Train the model.
- `evaluation`: Evaluate the trained model.

Before going on next step, let's review some of the core concepts of ZenML:

- **Artifact store**: Artifacts are the data that power your experimentation and model training. It is steps that produce artifacts; An artifact store is where artifacts are stored. The pipeline steps may have created these artifacts, or they may be the data first ingested into a pipeline via an ingestion step.
- **Metadata store**: Metadata are the pieces of information tracked about the pipelines, experiments, and configurations that you are running with ZenML. Metadata are stored inside the metadata store.
- **Container registry**: Some orchestrators will require you to containerize the steps of your pipeline. A container registry is a store for these (Docker) containers. A ZenML workflow involving a container registry will containerize your code and store the resulting container in the registry.
- **Kubeflow orchestrator**: An orchestrator manages the running of each step of the pipeline, administering the actual pipeline runs. It controls how and where each step within a pipeline is executed.

### Run the pipeline on a local orchestrator

With all the installation and initialization out of the way, all that's left to do is configure our ZenML stack. For this example, the stack we create consists of the following four parts:

- The **local artifact store** stores step outputs on your hard disk.
- The **local orchestrator** is responsible for running your ZenML pipeline.

We can now run the pipelines by simply executing the Python script. To run the
data analysis pipeline:

```bash
python run_kubeflow_pipeline.py analyze
```

To run the model training pipeline:

```bash
python run_kubeflow_pipeline.py train
```

## Run the same pipeline on Kubeflow Pipelines deployed to AWS

This stack has all components running in the AWS cloud:

* an AWS S3 artifact store
* a Kubeflow orchestrator installed in an AWS EKS Kubernetes cluster
* a metadata store that uses the same database as the Kubeflow deployment as
a backend
* an AWS ECR container registry
* a Seldon Core model deployer pointing to the AWS EKS cluster

### Setup Infrastructure with ZenML Stack recipes:

With [ZenML Stack Recipes](https://github.com/zenml-io/mlops-stacks), you can now provision all the infrastructure you need to run your ZenML pipelines with just a few simple commands.

The flow to get started for this example can be the following:

1. Pull this recipe to your local system.

    ```shell
    zenml stack recipe pull aws-minimal
    ```
2. 🎨 Customize your deployment by editing the default values in the `locals.tf` file.

3. 🔐 Add your secret information like keys and passwords into the `values.tfvars.json` file which is not committed and only exists locally.

4. 🚀 Deploy the recipe with this simple command.

    ```
    zenml stack recipe deploy aws-minimal
    ```

    > **Note**
    > If you want to allow ZenML to automatically import the created resources as a ZenML stack, pass the `--import` flag to the command above. By default, the imported stack will have the same name as the stack recipe, and you can provide your own with the `--stack-name` option.

5. You'll notice that a ZenML stack configuration file gets created after the previous command executes 🤯! This YAML file can be imported as a ZenML stack manually by running the following command.

    ```
    zenml stack import <stack-name> <path-to-the-created-stack-config-yaml>
    ```

> **Note**
>
>  You need to have your AWS credentials saved locally under ~/.aws/credentials

After you fulfill the prerequisites, we can configure ZenML.

Let's begin by setting up some environment variables to be used in the next steps:

```bash
zenml stack set aws
zenml stack up
```

When the setup is finished, you should see a new local URL that you can access
in your browser and take a look at the remote Kubeflow Pipelines UI (something
like http://localhost:8080).

### Running the Pipeline

```bash
python run_kubeflow_pipeline.py train
```

Now, you can go to [the localhost URL](http://localhost:8080/#/runs) to see the
UI (note that your port value may differ).

We can fetch the model from Kubeflow Pipelines and use it in our Inference pipeline.
We'll do a variation of this in the next chapter that also serves the model with
Seldon Core. The following diagram shows the flow of the whole pipeline:
![cloudkubeflowstack](_assets/wholekubeflowstack.gif)

## Continuous model deployment with Seldon Core

While building the real-world workflow for predicting whether a customer will churn or not, you might not want to train the model once and deploy it to production. Instead, you might want to train the model and deploy it to production when something gets triggered. This is where one of our recent integrations is valuable: [Seldon Core](https://docs.zenml.io/stack-components/model-deployers/seldon).

[Seldon Core](https://github.com/SeldonIO/seldon-core) is a production-grade open-source model serving platform. It packs a wide range of features built around deploying models to REST/GRPC microservices, including monitoring and logging, model explainers, outlier detectors, and various continuous deployment strategies such as A/B testing and canary deployments, and more.

In this project, we build a continuous deployment pipeline that trains a model and then serves it with Seldon Core as the industry-ready model deployment tool of choice. If you are interested in learning more about Seldon Core, you can check out our [docs](https://docs.zenml.io/stack-components/model-deployers/seldon). The following diagram shows the flow of the whole pipeline:
![seldondeployment](_assets/seldoncondeploy.gif)


Running the Seldon Pipeline.

```shell
python run_seldon_deployment_pipeline.py --deploy
```

You can control which pipeline to run by passing the `--deploy` and the `--predict` flag to the `run_seldon_deployment_pipeline.py` launcher. If you run the pipeline with the `--deploy` flag, the pipeline will train the model and deploy if the model meets the evaluation criteria and then Seldon Core will serve the model for inference. If you run the pipeline with the `--predict` flag, this tells the pipeline only to run the inference pipeline and not the training pipeline.

You can also set the `--min-accuracy` to control the evaluation criteria.

Now, you can go to the [localhost:8080](http://localhost:8080/#/runs) to see the
UI (same as the previous step, your port value may differ).

You can also list the list of models served with Seldon Core by running
`zenml model-deployer models list` and inspect them with `zenml model-deployer models describe`.
For example:

```bash
$ zenml model-deployer models list
┏━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━┓
┃ STATUS │ UUID                                 │ PIPELINE_NAME                  │ PIPELINE_STEP_NAME         │ MODEL_NAME ┃
┠────────┼──────────────────────────────────────┼────────────────────────────────┼────────────────────────────┼────────────┨
┃   ✅   │ 3ef6c58b-793d-4f85-8edd-aad961717f90 │ continuous_deployment_pipeline │ seldon_model_deployer_step │ model      ┃
┗━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━┛

$ zenml model-deployer models describe 3ef6c58b-793d-4f85-8edd-aad961717f90
                                                             Properties of Served Model 3ef6c58b-793d-4f85-8edd-aad961717f90                                                              
┏━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ MODEL SERVICE PROPERTY │ VALUE                                                                                                                                                         ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ MODEL_NAME             │ model                                                                                                                                                         ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ MODEL_URI              │ s3://zenml-projects/seldon_model_deployer_step/output/2517/seldon                                                                                                   ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ PIPELINE_NAME          │ continuous_deployment_pipeline                                                                                                                                ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ PIPELINE_RUN_ID        │ continuous_deployment_pipeline-27_May_22-12_47_38_918889                                                                                                      ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ PIPELINE_STEP_NAME     │ seldon_model_deployer_step                                                                                                                                    ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ PREDICTION_URL         │ http://abb84c444c7804aa98fc8c097896479d-377673393.us-east-1.elb.amazonaws.com/seldon/kubeflow/zenml-3ef6c58b-793d-4f85-8edd-aad961717f90/api/v0.1/predictions ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ SELDON_DEPLOYMENT      │ zenml-3ef6c58b-793d-4f85-8edd-aad961717f90                                                                                                                    ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ STATUS                 │ ✅                                                                                                                                                            ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ STATUS_MESSAGE         │ Seldon Core deployment 'zenml-3ef6c58b-793d-4f85-8edd-aad961717f90' is available                                                                              ┃
┠────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┨
┃ UUID                   │ 3ef6c58b-793d-4f85-8edd-aad961717f90                                                                                                                          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

```

## 🕹 Demo App

We built two Streamlit applications for our two different deployment solutions: one which fetches the pipeline from Kubeflow Pipelines and one which fetches the pipeline from the Seldon Core model service. You can view the Streamlit web application [here](https://share.streamlit.io/ayush714/zenfiles/customer-churn/customer-churn/streamlit_app_kubeflow.py).

You can run the following command to run the Streamlit application for the Kubeflow deployment:

```bash
streamlit run streamlit_app_kubeflow.py
```

You can run the following command to run the Streamlit application for the Seldon deployment:

```bash
streamlit run streamlit_app_seldon.py
```
