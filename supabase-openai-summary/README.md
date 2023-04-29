# Using GPT and OpenAI to generate a daily summary of a Supabase Database

Large language models (LLMs) have become a cornerstone of natural language processing, offering unparalleled capabilities for knowledge generation and reasoning. The past few weeks have seen a number of high profile releases of models and interfaces. However, despite their immense potential, incorporating custom, private data into these models [remains a challenge](https://docs.google.com/presentation/d/1VXQkR65ieROCmJP_ga09gGt8wkTGtTAdvaDRxMB67GI/edit#slide=id.p). 


## Installation

Install the required packages via the `requirements.txt` file located in the
`/src` directory.

```bash
pip install -r src/requirements.txt
```

## Connect to your deployed ZenML

```bash
zenml connect --url https://*** --username *** --password ***
```

## Create the secrets

You need to create the following secrets:

```bash
zenml secret create supabase \
    --supabase_url=<YOUR_SUPABASE_URL> \
    --supabase_key=<YOUR_SUPABASE_KEY>

zenml secret create openai --api_key=$OPENAPI_API_KEY   
```

## Running it locally

After the installation is completed, you directly run the pipeline locally
right away.

```bash
python run.py
```

Note that the pipeline is configured to generate artifacts relating to ZenML's
documentation and examples.

## Running it on a remote stack with an alerter

You can also run it on a stack with a remote artifact store and a Slack alerter:

```bash
zenml integration install gcp slack -y
```

```
zenml artifact-store register gcp_store -f gcp --path=gs://PATH_TO_STORE

zenml alerter register slack_alerter -f slack --slack_token=xoxb-252073111237684-3578623123400484-eeHtdsdfdfacK8ZJk20pHhamV --default_slack_channel_id=C03EDA8X0X

zenml stack register -a gcp_store -o default --alerter=slack_alerter --active
```