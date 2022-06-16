# UW-Madison GI Tract Image Segmentation

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

**Problem statement**: We will create a model to automatically segment the stomach and intestines on Magnetic resonance imaging (MRI) scans. We will be using data from [UW-Madison GI Tract Image Segmentation Competiton](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data) to build our model. In this competition we are asked segment organ cells in images of the stomach and intestines. If you'd like to know more about the problem statement, please visit the [competiton page](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation).

The training annotations are provided as RLE-encoded masks, and the images are in 16-bit grayscale PNG format. Training data has several cases for each image, and each case has a different set of annotations.

Our aim is not to win this competition, but to show you the power of the ZenML that how it can ease the whole process with it's amazing features like caching, easily switching stacks, training on different cloud service provider, and so on. The purpose of this repository is to demonstrate how ZenML empowers your business to build and deploy machine learning pipelines even on computer vision task in a multitude of ways:

- By offering you a framework or template to develop within.
- By Integrating with popular tools like `wandb` for experiment tracking.
- By using amazing features of ZenML like caching, training on cloud using step operators, and so on.

## :snake: Python Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenfiles.git
cd zenfiles/image-segmentation
pip install -r requirements.txt
```

We also need to install the following ZenML integrations for this project:

```bash
zenml integration install -y wandb
```

## 📙 Resources & References

Make sure to read the blog that introduces this project in detail: [Segmenting stomach and Intestines in MRI Scan](#).

# :thumbsup: The Solution

In order to build a model which will segment stomach and intestine from MRI scans & setting this in real-world workflow, we will build a reproducible pipeline using ZenML for this task, we will be using step operators for training on cloud (I will be using AWS but feel free to choose your favorite cloud provider.), we will also make use of ZenML's wandb integration for experiment tracking.

Our training pipeline `run_image_seg_pipeline.py` will be built using the following steps:-

- `prepare_df`: This step will read the data and prepare it for the pipeline.
- `create_stratified_fold`: This step creates stratified k folds.
- `augment_df`: This is a step that returns a dictionary of data transforms( the transformation which we need to apply on our data).
- `prepare_dataloaders`: This step takes in the dataframe and the data transforms and returns the train and validation dataloaders.
- `initiate_model_and_optimizer`: This is a step that returns (U-Net model, Adam optimizer, Configured scheduler).
- `train_model`: a step that takes in the model, optimizer, scheduler, train_loader, and valid_loader and returns the trained model and history.

We need to train the models on remote environment, so we need to use `StepOperator` to run your training jobs on remote backends. For this project, we will be using sagemaker as our remote backend to run our training jobs. We have several types of `StepOperator` and each type of step operator has their own prerequisites. Before running this project, you must set up the individual cloud providers in a certain way. The complete guide can be found in the [docs](https://docs.zenml.io/advanced-guide/cloud/step-operators).

Let's first create a sagemaker stack, you can create it by following commands:-

```bash
# install ZenML integrations
zenml integration install aws s3

zenml artifact-store register s3_store \
    --flavor=s3 \
    --path=<S3_BUCKET_PATH>

# create the sagemaker step operator
zenml step-operator register sagemaker \
    --flavor=sagemaker \
    --role=<SAGEMAKER_ROLE> \
    --instance_type=<SAGEMAKER_INSTANCE_TYPE>
    --base_image=<CUSTOM_BASE_IMAGE>
    --bucket_name=<S3_BUCKET_NAME>
    --experiment_name=<SAGEMAKER_EXPERIMENT_NAME>

# register the container registry
zenml container-registry register ecr_registry --flavor=aws --uri=<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# register and activate the sagemaker stack
zenml stack register sagemaker_stack \
    -m default \
    -o default \
    -c ecr_registry \
    -a s3_store \
    -s sagemaker \
    --set
```

If you have other cloud service provider like Azure or GCP, feel free to visit [this example](https://github.com/zenml-io/zenml/tree/main/examples/step_operator_remote_training) for setting up the stack for different remote backend.

We also need to integrate Weights & Biases tracking into our pipeline, we can create the stack with the wandb experiment tracker component by following command:

```bash
zenml experiment-tracker register wandb_tracker --type=wandb \
    --api_key=<WANDB_API_KEY> \
    --entity=<WANDB_ENTITY> \
    --project_name=<WANDB_PROJECT_NAME>
```

Now we can register a new stack with our experiment tracker component using the following command:

```bash
zenml stack register sagemaker_stack_with_wandb \
    -m default \
    -o default \
    -c ecr_registry \
    -a s3_store \
    -s sagemaker \
    -e wandb_tracker \
    --set
```

We created a stack named `sagemaker_stack_with_wandb` which has `StepOperator` component as sagemaker, and `wandb` as experiment tracker.

Now you can run the `run_image_seg_pipeline.py` by the following command: 

```bash 
python run_image_seg_pipeline.py
``` 
