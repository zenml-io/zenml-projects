# Guide:

1- Create a virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2- Install ZenML:
```bash
pip install zenml
```

3- Make a ZenML Cloud account and create a new tenant

4- Connect your local ZenML to the cloud:

5- Install the requirements and integrations:
```bash
zenml integration install mlflow sklearn evidently bentoml
```

6- Register the local stack components and local stack:
```bash
zenml experiment-tracker register -f mlflow local_mlflow_tracker && \ 
zenml data-validator register -f evidently evidently_data_validator && \
zenml model-deployer register bentoml_deployer --flavor=bentoml && \
zenml stack register -a default -o default -e local_mlflow_tracker -dv evidently_data_validator -d bentoml_deployer local_stack && \
zenml stack set local_stack
```

7- Let's run the pipeline:
```bash
python run.py --training
```

8- Promotion of the model:

9- Let's deploy the model:
```bash
python run.py --deployment
```