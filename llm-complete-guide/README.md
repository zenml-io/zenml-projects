# 🦜 Production-ready RAG pipelines for chat applications

This project showcases how you can work up from a simple RAG pipeline to a more
complex setup that involves finetuning embeddings, reranking retrieved
documents, and even finetuning the LLM itself. We'll do this all for a use case
relevant to ZenML: a question answering system that can provide answers to
common questions about ZenML. This will help you understand how to apply the
concepts covered in this guide to your own projects.

![](.assets/rag-pipeline-zenml-cloud.png)

Contained within this project is all the code needed to run the full pipelines.
You can follow along [in our
guide](https://docs.zenml.io/user-guides/llmops-guide/) to understand the
decisions and tradeoffs behind the pipeline and step code contained here. You'll
build a solid understanding of how to leverage LLMs in your MLOps workflows
using ZenML, enabling you to build powerful, scalable, and maintainable
LLM-powered applications.

This project contains all the pipeline and step code necessary to follow along
with the guide. You'll need a vector store to store the embeddings; full
instructions are provided below for how to set that up.

## 📽️ Watch the webinars

We've recently been holding some webinars about this repository and project. Watch the videos below if you want an introduction and context around the code and ideas covered in this project.

[![Building and Optimizing RAG Pipelines: Data Preprocessing, Embeddings, and Evaluation with ZenML](https://github.com/user-attachments/assets/1aea2bd4-8079-4ea2-98e1-8da6ba9aeebe)](https://www.youtube.com/watch?v=PazRMY8bo3U)

## 🏃 How to run

This project showcases production-ready pipelines so we use some cloud
infrastructure to manage the assets. You can run the pipelines locally using a
local PostgreSQL database, but we encourage you to use a cloud database for
production use cases.

### Setup your environment

Make sure you're running from a Python 3.8+ environment. Setup a virtual
environment and install the dependencies using the following command:

```shell
pip install -r requirements.txt
```

Depending on your hardware you may run into some issues when running the `pip install` command with the
`flash_attn` package. In that case running `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation` 
could help you. Possibly you might also need to install torch separately.

In order to use the default LLM for this query, you'll need an account and an
API key from OpenAI specified as a ZenML secret:

```shell
zenml secret create llm-complete --openai_api_key=<your-openai-api-key>
export ZENML_PROJECT_SECRET_NAME=llm-complete
```

### Setting up Pinecone

[Pinecone](https://www.pinecone.io/) is the default vector store used in this project. It's a cloud-native vector database that's optimized for machine learning applications. You'll need to create a Pinecone account and get an API key to use it.

Once you have your Pinecone account set up, you'll need to store your API key and index name as a ZenML secret. You can do this by running the following command:

```shell
zenml secret update llm-complete -v '{"pinecone_api_key": "YOUR_PINECONE_API_KEY", "pinecone_env": "YOUR_PINECONE_ENV", "pinecone_index": "YOUR_INDEX_NAME"}'

```

The `pinecone_index` value you specify will be used for all your development pipeline runs. Make sure the value consists only of alphanumeric characters and dashes. When you promote your ZenML model to production and run your ingestion pipeline again, it will automatically create a new production index called `<YOUR_INDEX_NAME>-prod`. This separation ensures that your development and production environments remain isolated.

### Choosing Your Vector Store

While Pinecone is the default vector store, this project supports multiple vector stores. You can choose between:

1. **Pinecone** (default): A cloud-native vector database optimized for machine learning applications
2. **PostgreSQL with pgvector**: A local or cloud PostgreSQL database with vector similarity search capabilities
3. **Elasticsearch**: A distributed search engine with vector search support

To switch between vector stores, you need to create or modify a pipeline configuration file (e.g., `configs/dev/rag.yaml`) and set the `index_type` parameter for the `index_generator` step. For example:

```yaml
steps:
  index_generator:
    parameters:
        index_type: pinecone  # Options: pinecone, postgres, elasticsearch
```

This configuration will be used by both the basic RAG and RAG pipelines. Each vector store requires its own setup and credentials as described in their respective sections below.

### Alternative: Setting up Supabase

While Pinecone is the default vector store, you can still use Supabase's PostgreSQL database as an alternative. 

[Supabase](https://supabase.com/) is a cloud provider that offers a PostgreSQL
database. It's simple to use and has a free tier that should be sufficient for
this project. Once you've created a Supabase account and organization, you'll
need to create a new project.

![](.assets/supabase-create-project.png)

You'll want to save the Supabase database password as a ZenML secret so that it
isn't stored in plaintext. You can do this by running the following command:

```shell
zenml secret update llm-complete -v '{"supabase_password": "YOUR_PASSWORD", "supabase_user": "YOUR_USER", "supabase_host": "YOUR_HOST", "supabase_port": "YOUR_PORT"}'
```

You can get the user, host and port for this database instance by getting the connection
string from the Supabase dashboard.

![](.assets/supabase-connection-string.png)

In case neither Pinecone nor Supabase is an option for you, you can use a different database as the backend.

### Running the RAG pipeline

To run the pipeline, you can use the `run.py` script. This script will allow you
to run the pipelines in the correct order. You can run the script with the
following command:

```shell
python run.py rag
```

This will run the basic RAG pipeline, which scrapes the ZenML documentation and
stores the embeddings in your configured vector store (Pinecone by default).

### Querying your RAG pipeline assets

Once the pipeline has run successfully, you can query the assets in your vector store
using the `--query` flag as well as passing in the model you'd like to
use for the LLM.

Note that you'll need to set the `LANGFUSE_API_KEY` environment variable for the
tracing which is built in to the implementation of the inference. This will
trace all LLM calls and store them in the [Langfuse](https://langfuse.com/)
platform.

When you're ready to make the query, run the following command:

```shell
python run.py query --query-text "how do I use a custom materializer inside my own zenml steps? i.e. how do I set it? inside the @step decorator?" --model=gpt4
```

Alternative options for LLMs to use include:

- `gpt4`
- `gpt35`
- `claude3`
- `claudehaiku`

Note that Claude will require a different API key from Anthropic. See [the
`litellm` docs](https://docs.litellm.ai/docs/providers/anthropic) on how to set
this up.

### Deploying the RAG pipeline

![](.assets/huggingface-space-rag-deployment.png)

You'll need to update and add some secrets to make this work with your Hugging
Face account. To get your ZenML service account API token and store URL, you can
first create a new service account:

```bash
zenml service-account create <SERVICE_ACCOUNT_NAME>
```

For more information on this part of the process, please refer to the [ZenML
documentation](https://docs.zenml.io/concepts/service_connectors).

Once you have your service account API token and store URL (the URL of your
deployed ZenML tenant), you can update the secrets with the following command:

```bash
zenml secret update llm-complete --zenml_api_token=<YOUR_ZENML_SERVICE_ACCOUNT_API_TOKEN> --zenml_store_url=<YOUR_ZENML_STORE_URL>
```

To set the Hugging Face user space that gets used for the Gradio app deployment,
you should set an environment variable with the following command:

```bash
export ZENML_HF_USERNAME=<YOUR_HF_USERNAME>
export ZENML_HF_SPACE_NAME=<YOUR_HF_SPACE_NAME> # optional, defaults to "llm-complete-guide-rag"
```

To deploy the RAG pipeline, you can use the following command:

```shell
python run.py deploy
```

This will open a Hugging Face space in your browser where you can interact with
the RAG pipeline.

### Run the LLM RAG evaluation pipeline

To run the evaluation pipeline, you can use the following command:

```shell
python run.py evaluation
```

You'll need to have first run the RAG pipeline to have the necessary assets in
the database to evaluate.

## RAG evaluation with Langfuse

You can run the Langfuse evaluation pipeline if you have marked some of your
responses as good or bad in the deployed Hugging Face space.

To run the evaluation pipeline, you can use the following command:

```shell
python run.py langfuse_evaluation
```

Note that this pipeline will only work if you have set the `LANGFUSE_API_KEY`
environment variable. It will use this key to fetch the traces from Langfuse and
evaluate the responses.

## Embeddings finetuning

For embeddings finetuning we first generate synthetic data and then finetune the
embeddings. Both of these pipelines are described in [the LLMOps guide](https://docs.zenml.io/v/docs/user-guides/llmops-guide/finetuning-embeddings) and
instructions for how to run them are provided below.

### Run the `distilabel` synthetic data generation pipeline

To run the `distilabel` synthetic data generation pipeline, you can use the following commands:

```shell
pip install -r requirements-argilla.txt # special requirements
python run.py synthetic
```

You will also need to have set up and connected to an Argilla instance for this
to work. Please follow the instructions in the [Argilla
documentation](https://docs.v1.argilla.io/en/latest/)
to set up and connect to an Argilla instance on the Hugging Face Hub. [ZenML's
Argilla integration
documentation](https://docs.zenml.io/v/docs/stack-components/annotators/argilla)
will guide you through the process of connecting to your instance as a stack
component.

Please use the secret from above to track all the secrets. Here we are also
setting a Huggingface write key. In order to make the rest of the pipeline work for you, you
will need to change the hf repo urls to a space you have permissions to.

```bash
zenml secret update llm-complete -v '{"argilla_api_key": "YOUR_ARGILLA_API_KEY", "argilla_api_url": "YOUR_ARGILLA_API_URL", "hf_token": "YOUR_HF_TOKEN"}'
```

### Finetune the embeddings

As with the previous pipeline, you will need to have set up and connected to an Argilla instance for this
to work. Please follow the instructions in the [Argilla
documentation](https://docs.v1.argilla.io/en/latest/)
to set up and connect to an Argilla instance on the Hugging Face Hub. [ZenML's
Argilla integration
documentation](https://docs.zenml.io/v/docs/stack-components/annotators/argilla)
will guide you through the process of connecting to your instance as a stack
component.

The pipeline assumes that your argilla secret is stored within a ZenML secret called `argilla_secrets`. 
![Argilla Secret](.assets/argilla_secret.png)

To run the pipeline for finetuning the embeddings, you can use the following
commands:

```shell
pip install -r requirements-argilla.txt # special requirements
python run.py embeddings
```

*Credit to Phil Schmid for his [tutorial on embeddings finetuning with Matryoshka
loss function](https://www.philschmid.de/fine-tune-embedding-model-for-rag) which we adapted for this project.*

## ☁️ Running in your own VPC

The basic RAG pipeline will run using a local stack, but if you want to improve
the speed of the embeddings step you might want to consider using a cloud
orchestrator. Please follow the instructions in documentation on popular integrations (currently available for
[AWS](https://docs.zenml.io/stacks/popular-stacks/aws-guide),
[GCP](https://docs.zenml.io/stacks/popular-stacks/gcp-guide), and
[Azure](https://docs.zenml.io/stacks/popular-stacks/azure-guide)) to learn how
you can run the pipelines on a remote stack.

If you run the pipeline using a cloud artifact store, logs from all the steps as
well as assets like the visualizations will all be shown in the ZenML dashboard.

### BONUS: Connect to ZenML Pro

If you run the pipeline using ZenML Pro you'll have access to the managed
dashboard which will allow you to get started quickly. We offer a free trial so
you can try out the platform without any cost. Visit the [ZenML Pro
dashboard](https://cloud.zenml.io/) to get started.

You can also self-host the ZenML dashboard. Instructions are available in our
[documentation](https://docs.zenml.io/getting-started/deploying-zenml).

## 📜 Project Structure

The project loosely follows [the recommended ZenML project structure](https://docs.zenml.io/user-guides/best-practices/set-up-your-repository):

```
.
├── LICENSE                                             # License file
├── README.md                                           # Project documentation
├── __init__.py
├── constants.py                                        # Constants used throughout the project
├── materializers
│   ├── __init__.py
│   └── document_materializer.py                        # Document materialization logic
├── most_basic_eval.py                                  # Basic evaluation script
├── most_basic_rag_pipeline.py                          # Basic RAG pipeline script
├── notebooks
│   └── visualize_embeddings.ipynb                      # Notebook to visualize embeddings
├── pipelines
│   ├── __init__.py
│   ├── generate_chunk_questions.py                     # Pipeline to generate chunk questions
│   ├── llm_basic_rag.py                                # Basic RAG pipeline using LLM
│   └── llm_eval.py                                     # Pipeline for LLM evaluation
├── requirements.txt                                    # Project dependencies
├── run.py                                              # Main script to run the project
├── steps
│   ├── __init__.py
│   ├── eval_e2e.py                                     # End-to-end evaluation step
│   ├── eval_retrieval.py                               # Retrieval evaluation step
│   ├── eval_visualisation.py                           # Evaluation visualization step
│   ├── populate_index.py                               # Step to populate the index
│   ├── synthetic_data.py                               # Step to generate synthetic data
│   ├── url_scraper.py                                  # Step to scrape URLs
│   ├── url_scraping_utils.py                           # Utilities for URL scraping
│   └── web_url_loader.py                               # Step to load web URLs
├── structures.py                                       # Data structures used in the project
├── tests
│   ├── __init__.py
│   └── test_url_scraping_utils.py                      # Tests for URL scraping utilities
└── utils
    ├── __init__.py
    └── llm_utils.py                                    # Utilities related to the LLM
```

## 🙏🏻 Inspiration and Credit

The RAG pipeline relies on code from [this Timescale
blog](https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector/)
that showcased using PostgreSQL as a vector database. We adapted it for our use
case and adapted it to work with Supabase.
