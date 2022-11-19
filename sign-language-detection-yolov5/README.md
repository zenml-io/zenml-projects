# Detect and recognize the American Sign Language alphabet on real-time image using Yolov5 and ZenML

**Problem statement**: One of the most anticipated capibilities of Machine Learning and AI is to help people with disabilities. The deaf community cannot do what most of the population take for granted and are often placed in degrading situations due to these challenges they face every day, in this Zenfile (project) will see how computer vision can be utilized to create a model that can bridge the gap for the deaf and hard of hearing by learning American Sign Language and be able to understand the meaning of each sign.
To so This project will use ZenML to create a pipeline that will train a model to detect and recognize the American Sign Language alphabet on real-time image using Yolov5 MLFlow and Vertex AI Platform.

The purpose of this repository is to demonstrate how [ZenML](https://github.com/zenml-io/zenml) empowers the build, track and deploy a computer vision pipeline using some of the most popular tools in the industry.

- By offering you a framework and template to base your own work on.
- By using a custom code Object Detection algorithm called [Yolov5](https://github.com/ultralytics/yolov5)
- By integrating with tools like [MLflow](https://mlflow.org/) to track the hyperparameters and metrics of the model.
- By allowing you to train your model on [Google Vertex AI Platform](https://cloud.google.com/vertex-ai) with minimal effort.

Note : This project is based on [Interactive ABC's with American Sign Language](https://github.com/insigh1/Interactive_ABCs_with_American_Sign_Language_using_Yolov5)
the main difference is that this project is using ZenML to create a pipeline that will train a model to detect and recognize the American Sign Language alphabet on real-time image using Yolov5 MLFlow and Vertex AI Platform.

## :snake: Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenfiles.git
git submodule update --init --recursive
cd zenfiles/sign-language-detection-yolov5
pip install -r requirements.txt
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you 
to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to  [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/getting-started/deploying-zenml), but first you must install the optional dependencies for the ZenML server:

```bash
zenml connect --url=$ZENML_SERVER_URL
zenml init
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
``` 
The zenfile can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## 📙 Resources & References

We had written a blog that explains this project in-depth: 

If you'd like to watch the video that explains the project, you can watch the [video](https://youtu.be/L3_pFTlF9EQ).

## :thumbsup: The Solution



### Training Pipeline

Our standard training pipeline consists of several steps:



### Deployment Pipeline


## :notebook: Diving into the code


## :question: FAQ

