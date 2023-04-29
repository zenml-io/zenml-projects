# Generate Daily Summary of Supabase Database using GPT-4 and ZenML

This project demonstrates how to create a daily summary of a Supabase database using OpenAI GPT-4 and ZenML. We use the YouTube video titles from [you-tldr](https://you-tldr.com) as an example and generate a summary of the last 24 hours of visitor activity. ZenML versions all data, allowing GPT-4 to compare the current summary to the previous one. The pipeline is executed daily using GitHub Actions and a deployed ZenML instance in Hugging Face Spaces.

The output of the pipeline posts the summary to a Slack channel:

![The summary in Slack](assets/youtldr_summarizer_slack.png)

## Installation

Install the required packages using the `requirements.txt` file in the `/src` directory:

```bash
pip install -r src/requirements.txt
```

## Connect to Your Deployed ZenML

Establish a connection with your deployed ZenML instance:

```bash
zenml connect --url https://*** --username *** --password ***
```

## Create Secrets

Create the necessary secrets for Supabase and OpenAI:

```bash
zenml secret create supabase \
    --supabase_url=$SUPABASE_URL \
    --supabase_key=$SUPABASE_KEY

zenml secret create openai --api_key=$OPENAPI_API_KEY   
```

## Run the Pipeline Locally

Once the installation is complete, you can run the pipeline locally:

```bash
python run.py
```

Note that the pipeline generates artifacts related to ZenML's documentation and examples by default.

## Run the Pipeline on a Remote Stack with Alerter

To run the pipeline on a remote stack with an artifact store and a Slack alerter, follow these steps:

1. Install the GCP and Slack integrations for ZenML:

```bash
zenml integration install gcp slack -y
```

2. Register the GCP artifact store:

```bash
zenml artifact-store register gcp_store -f gcp --path=gs://PATH_TO_STORE
```

3. Register the Slack alerter:

```bash
zenml alerter register slack_alerter -f slack --slack_token=xoxb-252073111237684-3578623123400484-eeHtdsdfdfacK8ZJk20pHhamV --default_slack_channel_id=C03EDA8X0X
```

4. Register the stack with the GCP artifact store and Slack alerter:

```bash
zenml stack register -a gcp_store -o default --alerter=slack_alerter --active
```

Once the stack is registered and set active, the pipeline will run on the remote stack with the GCP artifact store and send alerts to the specified Slack channel.

## Automate Pipeline Execution with GitHub Actions

To automate the pipeline execution every day, you can use GitHub Actions. First, store your secrets in the GitHub repository's secrets settings. Add the following secrets:

- `GCP_SA_KEY`: Your GCP service account key in JSON format.
- `ZENML_URL`: The URL of your deployed ZenML instance.
- `ZENML_USERNAME`: The username for your deployed ZenML instance.
- `ZENML_PASSWORD`: The password for your deployed ZenML instance.
- `ZENML_STACK`: The name of the ZenML stack you registered earlier.

Next, create a `.github/workflows/main.yml` file in your project and add the following content:

```yaml
name: Daily Summary

on:
  schedule:
    - cron: '0 0 * * *'

jobs:
  run_pipeline:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8

    - name: Log into GCP
      uses: 'google-github-actions/auth@v1'
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r src/requirements.txt
        zenml init
        zenml integration install gcp slack -y
        zenml connect --url ${{ secrets.ZENML_URL }} --username ${{ secrets.ZENML_USERNAME }} --password ${{ secrets.ZENML_PASSWORD }}
        zenml stack set ${{ secrets.ZENML_STACK }}
        python src/run.py

    - name: Run pipeline
      run: |
        python run.py
```

This configuration runs the pipeline every day at midnight. The workflow checks out the repository, sets up Python, logs into GCP, installs dependencies, connects to the deployed ZenML instance, sets the ZenML stack, and runs the pipeline.