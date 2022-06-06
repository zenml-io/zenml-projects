# 🍃 :arrow_right: 🔌 Predict electricity power generation based on wind forecast in Orkney, Scotland 

By its nature, renewable energy is highly weather-dependent, and the ongoing expansion of renewables is making our global power supply more vulnerable to changing weather conditions. Predicting how much power will be generated based on the weather forecast might be crucial, especially for areas such as Orkney in Scotland.


In this repository I showcase how to:
- Build a retrainable `zenml` pipeline
- Feature engineering - build numerical 2-dimensional vectors from the corresponding wind cardinal directions 
- Load data from Google Cloud BigQuery as a part of a `zenml` pipeline
- In the pipeline train your model remotely in Google Cloud Vertex AI


## 🐍 Python Requirements

Python dependencies:
```
git clone https://github.com/zenml-io/zenfiles.git
cd zenfiles/time-series-forecast
poetry install
```

`ZenML` integrations:
```
zenml integration install sklearn vertex gcp
```

Initialize zenml repository:
```
zenml init
```

## 👣  Step-by-Step on how to run the pipeline or reproduce this repository

I will show how to create google cloud resources for this project using `gcloud cli`, [follow this](https://cloud.google.com/sdk/docs/install) if you don't have it set up.

### 1. Make sure you are in the correct GCP project

```
gcloud config list
```

### 2. Set permissions to create and manage `Vertex AI` custom jobs and to access data from `BigQuery`

Create a service account
```
gcloud iam service-accounts create <NAME>

#EXAMPLE
gcloud iam service-accounts create zenml-sa
```

Grant permission to the service account ([list](https://cloud.google.com/bigquery/docs/access-control) of BQ roles)
```
gcloud projects add-iam-policy-binding <PROJECT_ID> --member="serviceAccount:<SA-NAME>@<PROJECT_ID>.iam.gserviceaccount.com" --role=<ROLE>

#EXAMPLE
gcloud projects add-iam-policy-binding zenml-vertex-ai --member="serviceAccount:zenml-sa@zenml-vertex-ai.iam.gserviceaccount.com" --role=roles/storage.admin
gcloud projects add-iam-policy-binding zenml-vertex-ai --member="serviceAccount:zenml-sa@zenml-vertex-ai.iam.gserviceaccount.com" --role=roles/aiplatform.admin
gcloud projects add-iam-policy-binding zenml-vertex-ai --member="serviceAccount:zenml-sa@zenml-vertex-ai.iam.gserviceaccount.com" --role=roles/bigquery.admin

```
Generate a key file
```
gcloud iam service-accounts keys create <FILE-NAME>.json --iam-account=<SA-NAME>@<PROJECT_ID>.iam.gserviceaccount.com

EXAMPLE
gcloud iam service-accounts keys create credentials.json --iam-account=zenml-sa@zenml-vertex-ai.iam.gserviceaccount.com
```
Set the environment variable:

To use service accounts with the Google Cloud CLI, you need to set an environment variable where your code runs
```
export GOOGLE_APPLICATION_CREDENTIALS=<KEY-FILE-LOCATION>
```
For the bigquery step you also need to point to the same file
```python

class BigQueryImporterConfig(BaseStepConfig):
    query: str = 'SELECT * FROM `computas_dataset.wind_forecast`'
    project_id: str = 'computas-project-345810'

@step
def bigquery_importer(config: BigQueryImporterConfig) -> pd.DataFrame:
    credentials = service_account.Credentials.from_service_account_file('credentials.json')
    return pandas_gbq.read_gbq(config.query, project_id = config.project_id, credentials = credentials)
```
NOTE: You also need to change the query and your project ID accordingly.

### 3. Create a GCP bucket

Vertex AI and zenml will use this bucket for output of any artifacts from the training run

```
gsutil mb -l <REGION> gs://bucket-name

#EXAMPLE
gsutil mb -l europe-west1 gs://zenml-bucket
```

### 4. Configure and enable container registry in GCP

This registry will be used by ZenML to push your job images that Vertex will use.

a) [Enable](https://cloud.google.com/container-registry/docs) Container Registry


b) [Authenticate](https://cloud.google.com/container-registry/docs/advanced-authentication) your local `docker` cli with your GCP container registry 

```
docker pull busybox
docker tag busybox gcr.io/<PROJECT-ID/busybox
docker push gcr.io/<PROJECT-ID>/busybox
```

### 5. [Enable](https://console.cloud.google.com/marketplace/product/google/aiplatform.googleapis.com?q=search&referrer=search&project=cloudguru-test-project) `Vertex AI API`

### 6. Build a custom image from `zenml` that will be used in the vertex step operator

```
cd src
docker build --tag zenmlcustom:0.1
```

### 7. Set up the components required for `zenml` stack

Set the bucket created earlier
```
zenml artifact-store register <NAME> --type=gcp --path=<GCS_BUCKET_PATH>

# EXAMPLE
zenml artifact-store register gcp-store --type=gcp --path=gs://zenml-bucket
```

Create the vertex step operator

```
zenml step-operator register <NAME> \
    --type=vertex \
    --project=<PROJECT-ID> \
    --region=<REGION> \
    --machine_type=<MACHINE-TYPE> \
    --base_image=<CUSTOM_BASE_IMAGE> #this can be left out if you wish to use zenml's default image

# EXAMPLE
zenml step-operator register vertex \
    --type=vertex \
    --project=zenml-vertex-ai \
    --region=europe-west1 \
    --machine_type=n1-standard-4 \
    --base_image=zenmlcustom:0.1
```

List of [available machines](https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types)

Register the container registry

```
zenml container-registry register <NAME> --type=default --uri=gcr.io/<PROJECT-ID>/<IMAGE>

#EXAMPLE
zenml container-registry register gcr_registry --type=default --uri=gcr.io/zenml-vertex-ai/busybox
```

Register the new stack (change names accordingly)
```
zenml stack register vertex_training_stack \
    -m default \
    -o default \
    -c gcr_registry \
    -a gcp-store \
    -s vertex
```

View all your stacks: `zenml stack list`

Activate the stack
```
zenml stack set vertex_training_stack
```

### ▶️ Run the Code

Now we're ready. Execute:

```bash
python main.py
```

# 📜 Useful links

https://docs.zenml.io/features/step-operators

https://docs.zenml.io/features/guide-aws-gcp-azure

https://cloud.google.com/docs/authentication/getting-started#create-service-account-gcloud

https://apidocs.zenml.io/0.7.3/cli/

https://blog.zenml.io/step-operators-training/
