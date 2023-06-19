# Building and Using an MLOps Stack with ZenML

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) enables your machine
learning projects in a multitude of ways:

- By offering you a framework or template to develop within
- By seamlessly integrating into the tools you love and need
- By allowing you to easily switch orchestrators for your pipelines
- By bringing much-needed Zen into your machine learning

**ZenML** is an extensible, open-source MLOps framework to create production-ready machine learning pipelines. Built for
data scientists, it has a simple, flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions that
are catered towards ML workflows.

At its core, **ZenML pipelines execute ML-specific workflows** from sourcing data to splitting, preprocessing, training,
all the way to the evaluation of results and even serving. There are many built-in batteries to support common ML
development tasks. ZenML is not here to replace the great tools that solve these individual problems. Rather, it
**integrates natively with popular ML tooling** and gives standard abstraction to write your workflows.

Within this repo, we will use ZenML to build pipelines that seamlessly use [Evidently](https://evidentlyai.com/),
[MLFlow](https://mlflow.org/), [Kubeflow Pipelines](https://www.kubeflow.org/) and post
results to our [Discord](https://discord.com/).

![img](./_assets/nba.png)

[![](https://img.youtube.com/vi/Ne-dt9tu11g/0.jpg)](https://www.youtube.com/watch?v=Ne-dt9tu11g)

_Come watch along as Hamza Tahir, Co-Founder and CTO of ZenML showcase an early version of this repo
to the MLOps.community._

## :computer: System Requirements

In order to run this demo you need to have some packages installed on your machine.

Currently, this will only run on UNIX systems.

| package | MacOS installation                                                               | Linux installation                                                                 |
| ------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| docker  | [Docker Desktop for Mac](https://docs.docker.com/desktop/mac/install/)           | [Docker Engine for Linux ](https://docs.docker.com/engine/install/ubuntu/)         |
| kubectl | [kubectl for mac](https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/) | [kubectl for linux](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/) |
| k3d     | [Brew Installation of k3d](https://formulae.brew.sh/formula/k3d)                 | [k3d installation linux](https://k3d.io/v5.2.2/)                                   |

## :snake: Python Requirements

Once you've got the system requirements figured out, let's jump into the Python packages you need.
Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenml-projects
cd zenml-projects/nba-pipeline
pip install -r requirements.txt
```

If you are running the `run_pipeline.py` script, install the following integrations by running the following commands in your terminal:

```bash
zenml integration install sklearn -y
zenml integration install aws -y
zenml integration install evidently -y
zenml integration install mlflow -y
zenml integration install kubeflow -y
```

## :basketball: The Task

A couple of weeks ago, we were looking for a fun project to work on for the next chapter of our ZenHacks. During our
initial discussions, we realized that it would be really great to work with an NBA dataset, as we could quickly get
close to a real-life application like a "3-Pointer Predictor" while simultaneously entertaining ourselves with one
of the trending topics within our team.

As we were building the dataset around a "3-Pointer Predictor", we realized that there is one factor that we need to
take into consideration first: Stephen Curry, The Baby Faced Assassin. In our opinion, there is no denying that he
changed the way that the games are played in the NBA and we wanted to actually prove that this was the case first.

That's why our story in this ZenHack will start with a pipeline dedicated to drift detection. As the breakpoint of this
drift, we will be using the famous "Double Bang" game that the Golden State Warriors played against Oklahoma City
Thunder back in 2016. Following that, we will build a training pipeline which will generate a model that predicts
the number of three-pointers made by a team in a single game, and ultimately, we will use these trained models and
create an inference pipeline for the upcoming matches in the NBA.

![Diagram depicting the Training and Inference pipelines](_assets/all_pipelines.png)

## :notebook: Diving into the code

We're ready to go now. You have two options:

### Notebook

You can spin up a step-by-step guide in `Building and Using An MLOPs Stack With ZenML.ipynb`:

```python
jupyter notebook
```

### Script

You can also directly run the code. First, set up the stack:

```shell
# You register an experiment tracker
zenml experiment-tracker register local_mlflow_tracker --flavor=mlflow

# You register a data validator
zenml data-validator register local_evidently --flavor=evidently

# Now it all is combined into the local_kubeflow_stack
zenml stack register local_stack -a default -o default -e local_mlflow_tracker -dv local_evidently

# And we activate the new stack, now all pipelines will be run within this stack
zenml stack set local_stack

# Check it out, your new stack is registered
zenml stack describe
```

Then, run the pipelines using `run_pipeline.py` script:

```python
python run_pipeline.py drift  # Run one-shot drift pipeline
python run_pipeline.py train  # Run training pipeline
python run_pipeline.py infer  # Run inference pipeline
```

## :rocket: From Local to Cloud Stack
In ZenML you can choose to run your pipeline on any infrastructure of your choice.
The configuration of the infrastructure is called a [Stack](https://docs.zenml.io/user-guide/starter-guide/understand-stacks). 
By switching the Stack, you can choose to run your pipeline locally or in the cloud.

In any Stack, there must be at least two basic [Stack Components](https://docs.zenml.io/user-guide/starter-guide/understand-stacks#components-of-a-stack): 
* [Orchestrator](https://docs.zenml.io/user-guide/starter-guide/understand-stacks#orchestrator) - Coordinates all the steps to run in a pipeline.
* [Artifact Store](https://docs.zenml.io/user-guide/starter-guide/understand-stacks#artifact-store) - Stores all data that pass through the pipeline. 

ZenML comes with a default local stack with a local orchestrator and local artifact store.
![local](_assets/local_cloud.png)


To transition from running our pipelines locally (see diagram above) to running them on Kubeflow Pipelines, we only need to register a new stack with some more stack components:

First, authenticate your credentials with:

```shell
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 715803424590.dkr.ecr.us-east-1.amazonaws.com
aws eks --region us-east-1 update-kubeconfig --name zenhacks-cluster --alias zenml-eks
```
Replace the `--region`, `--username`, `--password-stdin`, `--region`, `--name`, and `--alias` with your own.

Set the following environment variables with your namespace, username and password.
```
export KUBEFLOW_NAMESPACE="your-namespace"
export KUBEFLOW_USERNAME="yourusername@yours.io"
export KUBEFLOW_PASSWORD="yourpassword"
```

Now let's register all the stack components that we will be using.

MLflow Experiment Tracker:
```
zenml experiment-tracker register aws_mlflow_tracker  --flavor=mlflow --tracking_insecure_tls=true --tracking_uri="https://ac8e6c63af207436194ab675ee71d85a-1399000870.us-east-1.elb.amazonaws.com/mlflow" --tracking_username="{{mlflow_secret.tracking_username}}" --tracking_password="{{mlflow_secret.tracking_password}}" 
```

Evidently Data Validator:
```
zenml data-validator register evidently --flavor=evidently
```

Kubeflow Orchestrator:
```
zenml orchestrator register multi_tenant_kubeflow \
  --flavor=kubeflow \
  --kubernetes_context=zenml-eks \
  --kubeflow_hostname=https://www.kubeflowshowcase.zenml.io/pipeline
```
Make sure to pass in your own `--kubernetes_context` and `--kubeflow_hostname`.

Artifact Store on Amazon S3:
```
zenml artifact-store register s3_store -f s3 --path=s3://zenml-projects
```
Include your own `--path` on S3.

Container registry on ECR:
```
zenml container-registry register ecr_registry --flavor=aws --uri=715803424590.dkr.ecr.us-east-1.amazonaws.com 
```
Include your own `--uri` on ECR.

Finally, with all the stack components set up, let's register the stack:

```
zenml stack register kubeflow_gitflow_stack \
    -a s3_store \
    -c ecr_registry \
    -o multi_tenant_kubeflow \
    -e aws_mlflow_tracker \
    -dv evidently
```

Set the registered stack as the active stack
```
zenml stack set kubeflow_gitflow_stack
```

Let's register our secrets to the secrets store

```
zenml secret create mlflow_secret -i
```
You will be prompted to key in the `tracking_username` and `tracking_password` for MLflow.

Once all the configurations are done, you can provision the stack by:

```
zenml stack up
```

You are now ready to run the pipeline!

```python
python run_pipeline.py drift  # Run drift detection pipeline
python run_pipeline.py train  # Run training pipeline
python run_pipeline.py infer  # Run inference pipeline
```

And head over to your [Kubeflow central dashboard](https://www.kubeflow.org/docs/components/central-dash/overview/).


The drift detection pipeline looks like the following.
![kubeflow_dash](_assets/kubeflow_run_drift.png)


The training pipeline looks like the following.
![kubeflow_dash](_assets/kubeflow_run.png)

The inference pipeline looks like the following.
![kubeflow_dash](_assets/kubeflow_run_infer.png)


## :checkered_flag: Cleaning up when you're done

Once you are done running this notebook you might want to stop all running processes. For this, run the following command.
(This will tear down your `k3d` cluster and the local docker registry.)

```shell
zenml stack set kubeflow_gitflow_stack
zenml stack down -f
```
