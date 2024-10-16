# :: Track experiments in ZenML natively

Although ZenML plugs into many [experiment trackers](https://www.zenml.io/vs/zenml-vs-experiment-trackers), a lot of 
the functionality of experiment trackers is already covered by ZenML's native metadata and artifact tracking.
This project aims to show these capabilities.

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

## Explore your experiments

...




