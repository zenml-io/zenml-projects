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


#### Option 1 - Interactively explore the quickstart using Jupyter Notebook:
```bash
pip install notebook
jupyter notebook
# open quickstart.ipynb
```

#### Option 2 - Execute the whole ML pipeline from a Python script:
```bash
# Pip install all requirements
pip install -r requirements.txt

# Install required zenml integrations
zenml integration install sklearn pandas -y

# Initialize ZenML
zenml init
```

## 📈 Explore your experiments

...




