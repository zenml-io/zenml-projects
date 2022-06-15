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
