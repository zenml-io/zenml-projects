# Predicting whether the customer will churn or not before they even did it

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

**Problem statement**: For a given customer's historical data, we are asked to predict whether a customer will churn a company or not. We will be using [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?datasetId=13996&sortBy=voteCount) dataset for building an end to end production grade machine learning system that can predict whether the customer will churn or not. Dataset has 20 input features and a target variable for 7043 customers.

Customer churn is a tendency of customers to leave or churn a company and stop being a paying customer for a particular business. We can calculate a churn rate for a company by dividing the number of customers who churned by the total number of customers and then multiply that number by 100 percent. If you want to learn more about customer churn, you can read this[Wikipedia article](https://en.wikipedia.org/wiki/Churn_rate).

So, In order to achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict whether a customer will churn or not before they even did it.

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers your business to build and deploy machine learning pipelines in a multitude of ways:

- By offering you a framework or template to develop within.
- By integrating with popular tools like Kubeflow, Seldon-core, facets and more.
- By allowing you to build and deploy your machine learning pipelines easily using modern MLOps Framework.

## :snake: Python Requirements [WIP]

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenfiles.git
cd zenfiles/customer-churn
pip install -r requirements.txt
```

We need to install following integrations for this project:

```bash
zemml integration install mlflow -f
zemml integration install kubeflow -f
```

## 📙 Resources & References

We had written a blog that explains this project in-depth: [Predicting whether the customer will churn or not before they even did it](#).

If you'd like to watch the video that explains the project, you can watch the [video](#).

# :thumbsup: The Solution

We showcase two solutions to this problem:

- `Deployment using Kubeflow pipelines`: We will be using Kubeflow pipelines to build and run our ZenML pipeline on cloud and deploy it in a production environment.
- `Continuous Deployment using Seldon Core`: We will be using Seldon Core which is a production grade open source model serving platform to build our continuous deployment pipeline that trains a model and then serves it with Seldon Core.

## Deploy pipelines to production using Kubeflow pipelines

In order to build real world workflow for predicting whether a customer will churn or not, Most likely you will probably develop your pipelines on your local machine initially as this allows for quicker iteration and debugging. However, at a certain point when you are finished with its design, you might want to transition to a more production-ready setting and deploy the pipeline to a more robust environment. This is where ZenML comes in.

We will be using [Kubeflow](#) integration of ZenML for deploying pipelines to production using Kubeflow pipelines on cloud.

Our training pipeline `run_kubeflow_pipeline.py` will be built using the following steps:

- `ingest_data`: This step will be used to ingest the data from the source and create a DataFrame.
- `encode_cat_cols`: This step will be used to encode categorical columns.
- `handle_imbalanced_data`: This step will be used to handle imbalanced data.
- `drop_cols`: This step will be used to irrelevant drop columns.
- `data_splitter`: This step will be used to split the data into training and test sets.
- `model_trainer`: This step will be used to train the model.
- `evaluation`: This step will be used to evaluate the trained model.

#### Run the same pipeline on a local Kubeflow Pipelines deployment

Now with all the installation and initialization out of the way, all that's left to do is configuring our ZenML stack. For this example, the stack we create consists of the following four parts:

- The **local artifact store** stores step outputs on your hard disk.
- The **local metadata store** stores metadata like the pipeline name and step
  parameters inside a local SQLite database.
- The docker images that are created to run your pipeline are stored in a local
  docker **container registry**.
- The **Kubeflow orchestrator** is responsible for running your ZenML pipeline
  in Kubeflow Pipelines.

```bash
# Make sure to create the local registry on port 5000 for it to work
zenml container-registry register local_registry --type=default --uri=localhost:5000
zenml orchestrator register kubeflow_orchestrator --type=kubeflow
zenml stack register local_kubeflow_stack \
    -m local_metadata_store \
    -a local_artifact_store \
    -o kubeflow_orchestrator \
    -c local_registry

# Activate the newly created stack
zenml stack set local_kubeflow_stack
```

Now, we need to startup the Kubeflow pipelines locally, All we need to do is run:

```bash
zenml stack up
```

When the setup is finished, you should see a local URL which you can access in
your browser and take a look at the Kubeflow Pipelines UI.

We can now run the pipeline by simply executing the python script:

```bash
python run_kubeflow_pipeline.py
```

This will build a docker image containing all the necessary python packages and
files, push it to the local container registry and schedule a pipeline run in
Kubeflow Pipelines. Once the script is finished, you should be able to see the
pipeline run [here](http://localhost:8080/#/runs).

#### Run the same pipeline on Kubeflow Pipelines deployed to aws

We will now run the same pipeline in Kubeflow Pipelines deployed to a AWS EKS cluster. Prior to running this, you need some additional setup or prerequisites to run the pipeline on AWS, you can refer to our [documentation](https://docs.zenml.io/features/guide-aws-gcp-azure#pre-requisites) which will help you in fulfilling the requirement for running the pipeline on AWS.

If you want to run the pipeline on other cloud providers like GCP, Azure, you can follow [this guide](https://docs.zenml.io/features/guide-aws-gcp-azure) in order to run the pipeline on that cloud provider. Specifically for this project, we will be using AWS but feel free to use any cloud provider you want.

After you fulfill the prerequisites, now we need to Integrate with ZenML.

1. Install the cloud provider

```bash
zenml integration install aws
```

2. Register the stack components

```bash
zenml container-registry register cloud_registry --type=default --uri=$PATH_TO_YOUR_CONTAINER_REGISTRY
zenml orchestrator register cloud_orchestrator --type=kubeflow --custom_docker_base_image_name=YOUR_IMAGE
zenml metadata-store register kubeflow_metadata_store --type=kubeflow
zenml artifact-store register cloud_artifact_store --type=s3 --path=$PATH_TO_YOUR_BUCKET

# Register the cloud stack
zenml stack register cloud_kubeflow_stack -m kubeflow_metadata_store -a cloud_artifact_store -o cloud_orchestrator -c cloud_registry
```

3. Activate the newly created stack.

```bash
zenml stack set cloud_kubeflow_stack
```

4. Do a pipeline run and check your Kubeflow UI to see it running there! 🚀

```bash
python run_kubeflow_pipeline.py
```

## Continuous model deployment with Seldon Core [WIP]

While building the real world workflow for predicting whether a customer will churn or not, you might not want to train the model once and deploy it to production. Instead you might want to train the model and deploy it to production when something get triggered. This is where one of our recent Integration comes in, [Seldon Core](#).

[Seldon Core](#) is a production grade open source model serving platform. It packs a wide range of features built around deploying models to REST/GRPC microservices that include monitoring and logging, model explainers, outlier detectors and various continuous deployment strategies such as A/B testing, canary deployments and more.

It also comes equipped with a set of built-in model server implementations designed to work with standard formats for packaging ML models that greatly simplify the process of serving models for real-time inference.

In this project, we build a continuous deployment pipeline that trains a model and then serves it with Seldon Core as the industry-ready model deployment tool of choice.
