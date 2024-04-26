# Guide:

0- Clone this project

```bash
git clone --depth 1 --branch feature/youtube-video-e2e-example https://github.com/zenml-io/zenml-projects.git && cd zenml-projects/e2e
```

1- Create a virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2- Install ZenML and requirements:
```bash
pip install -r requirements.txt
```

4- Connect your local ZenML to the cloud:

```bash
zenml connect --url <YOUR_TENANT_URL>
```

5- Register the local stack components and local stack:
```bash
zenml experiment-tracker register -f mlflow local_mlflow_tracker && \ 
zenml data-validator register -f evidently evidently_data_validator && \
zenml model-deployer register bentoml_deployer --flavor=bentoml && \
zenml stack register -a default -o default -e local_mlflow_tracker -dv evidently_data_validator -d bentoml_deployer local_stack && \
zenml stack set local_stack
```

6- Let's run the pipeline:
```bash
python run.py --training
```

7- Promotion of the model:

8- Let's deploy the model:
```bash
python run.py --deployment
```