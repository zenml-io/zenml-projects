# End-to-end Computer Vision 🌄

This is a project that demonstrates an end-to-end computer vision pipeline using
ZenML. The pipeline is designed to be modular and flexible, allowing for easy
experimentation and extension.

The project showcases the full lifecycle of a computer vision project, from data
collection and preprocessing to model training and evaluation. The pipeline also
incorporates a human-in-the-loop (HITL) component, where human annotators can
label images to improve the model's performance, as well as feedback using
[Voxel51's FiftyOne](https://voxel51.com/fiftyone/) tool.

The project uses the [Ship Detection
dataset](https://huggingface.co/datasets/datadrivenscience/ship-detection) from
DataDrivenScience on the Hugging Face Hub, which contains images of ships in
satellite imagery. The goal is to train a model to detect ships in the images.
Note that this isn't something that our YOLOv8 model is particularly good at out
of the box, so it serves as a good example of how to build a pipeline that can
be extended to other use cases.

The project consists of the following pipelines and steps:

- 

## Run this pipeline

### Setup

You'll need to run the following:

```bash
zenml integration install label_studio torch gcp mlflow -y
pip install -r requirements.txt
pip uninstall wandb
```

You can also set the following environment variables:

```bash
export DATA_UPLOAD_MAX_NUMBER_FILES=1000000
export WANDB_DISABLED=True
```

And to use the Albumentations and annotation plugins, you'll need to install
them:

```bash
fiftyone plugins download https://github.com/jacobmarks/fiftyone-albumentations-plugin

fiftyone plugins download https://github.com/voxel51/fiftyone-plugins --plugin-names @voxel51/annotation
```




# Data

initial dataset is stored at:
`gs://zenml-internal-artifact-store/raw-ship-data/`

copy with: `gsutil -m cp -r gs://zenml-internal-artifact-store/raw-ship-data/
/path`
