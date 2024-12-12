# :: Track experiments in ZenML natively

Although ZenML plugs into many [experiment trackers](https://www.zenml.io/vs/zenml-vs-experiment-trackers), a lot of 
the functionality of experiment trackers is already covered by ZenML's native metadata and artifact tracking.
This project aims to show these capabilities.

## 🎯 Project Overview
We're tackling a simple classification task using the breast cancer dataset. Our goal is to showcase how ZenML can effortlessly track experiments, hyperparameters, and results throughout the machine learning workflow.

### 🔍 What We're Doing

In this project, we begin by preparing the breast cancer dataset for our model through data preprocessing. For our machine learning task, we've chosen to use an SGDClassifier. Rather than relying on sklearn's GridSearchCV, we implement our own hyperparameter tuning process to showcase ZenML's robust tracking capabilities. Finally, we conduct a thorough analysis of the results, visualizing how various hyperparameters influence the model's accuracy. This approach allows us to demonstrate the power of ZenML in tracking and managing the machine learning workflow.

We are by no means claiming that our solution outperforms GridSearchCV, spoiler alert, this demo won't, rather, this project demonstrates how you would do hyperparameter tuning and experiment tracking  with ZenML on large deep learning problems. 

### 🛠 The Pipeline

Our ZenML pipeline consists of the following steps:

The feature_engineering pipeline:
* Data Loading: Load the breast cancer dataset.
* Data Splitting: Split the data into training and testing sets.
* Data Pre Processing: Pre process our dataset

The model training pipeline:
* Model Training: Train multiple SGDClassifiers with different hyperparameters.
* Model Evaluation: Evaluate each model's performance.

By running this pipeline iteratively 

## :running: Run locally

```bash
# Pip install all requirements
pip install -r requirements.txt

# Install required zenml integrations
zenml integration install sklearn pandas -y

# Initialize ZenML
zenml init

# Connect to your ZenML server
zenml login ...

python run.py --parallel
```

This will run a grid search across the following parameter space:

```python
alpha_values = [0.0001, 0.001, 0.01]
penalties = ["l2", "l1", "elasticnet"]
losses = ["hinge", "squared_hinge", "modified_huber"]
```

If you choose to include the `--parallel` flag, this should all run in parallel. 
As ZenML smartly caches across pipelines, and because the feature pipeline has run 
ahead of the parallel training runs, all training pipelines should start on the
`model_trainer` step.
![Pipeline DAG with cached steps](./assets/pipeline_dag_caching.png)

After running, you now should have 27 runs of the model training with 27
produced model_versions. In case you are running with [ZenML Pro](https://docs.zenml.io/getting-started/zenml-pro)
you'll now be able to inspect these models in the dashboard:
![Model Versions Page](./assets/model_versions.png)

Additionally, in case you ran with a remote [Data backend](https://docs.zenml.io/stack-components/artifact-stores),
you'll be able to inspect the confusion matrix for any specific training directly in the
frontend.
![Confusion Matrix Visualization](./assets/cm_visualization.png)

In case you want to create your own visualization, check out the implementation
at `native-experiment-tracking/steps/model_trainer.py:generate_cm`. Basically, just create a 
matplotlib plot, convert it into a `PIL.Image` and return it from your
step. Don't forget to annotate your [step output accordingly](https://docs.zenml.io/how-to/build-pipelines/step-output-typing-and-annotation.

```python
from typing import Tuple
from typing_extensions import Annotated
from PIL import Image
from zenml import ArtifactConfig, step

@step
def func(...) -> Tuple[
    Annotated[
        ...
    ],
    Annotated[
        Image.Image, "confusion_matrix"
    ]
]:
```

## 📈 Explore your experiments

Once all pipelines ran, it is time to analyze our experiment.
For this we have written an analyze.py script.
```commandline
python analyze.py
```
This will generate 2 plots for you:

**3D Plot**
![3D Plot](./assets/3d_plot.png)

**2D Plot**
![2D Plot](./assets/2d_plot.png)

Feel free to use this file as a starting point to write your very own
analysis. 

## The moral of the story

So what's the point? We at ZenML believe that any good experiment should be set up in a
repeatable, scalable way while storing all the relevant metadata in order to analyze the experiment 
after the fact. This project shows how you could do this with ZenML. 

Once you have accomplished this on a toy dataset with a tiny SGDClassifier, you can start 
scaling up in all dimensions: data, parameters, model, etc... And all of this while staying infrastructure 
agnostic. So when your experiment outgrows your local machine, you can simply move 
to the stack of your choice ...

## 🤝 Contributing

Contributions to improve the pipeline are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
