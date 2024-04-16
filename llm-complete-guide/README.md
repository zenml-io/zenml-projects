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
guide](https://docs.zenml.io/user-guide/llmops-guide/) to understand the
decisions and tradeoffs behind the pipeline and step code contained here. You'll
build a solid understanding of how to leverage LLMs in your MLOps workflows
using ZenML, enabling you to build powerful, scalable, and maintainable
LLM-powered applications.

This project contains all the pipeline and step code necessary to follow along
with the guide. You'll need a PostgreSQL database to store the embeddings; full
instructions are provided below for how to set that up.

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

In order to use the default LLM for this query, you'll need an account and an
API key from OpenAI specified as another environment variable:

```shell
export OPENAI_API_KEY=<your-openai-api-key>
```

### Setting up Supabase

[Supabase](https://supabase.com/) is a cloud provider that provides a PostgreSQL
database. It's simple to use and has a free tier that should be sufficient for
this project. Once you've created a Supabase account and organisation, you'll
need to create a new project.

![](.assets/supabase-create-project.png)

You'll want to save the Supabase database password as a ZenML secret so that it
isn't stored in plaintext. You can do this by running the following command:

```shell
zenml secret create supabase_postgres_db --password="YOUR_PASSWORD"
```

You'll then want to connect to this database instance by getting the connection
string from the Supabase dashboard.

![](.assets/supabase-connection-string.png)

You can use these details to populate some environment variables where the
pipeline code expects them:

```shell
export ZENML_POSTGRES_USER=<your-supabase-user>
export ZENML_POSTGRES_HOST=<your-supabase-host>
export ZENML_POSTGRES_PORT=<your-supabase-port>
```

### Running the RAG pipeline

To run the pipeline, you can use the `run.py` script. This script will allow you
to run the pipelines in the correct order. You can run the script with the
following command:

```shell
python run.py --rag
```

This will run the basic RAG pipeline, which scrapes the ZenML documentation and
stores the embeddings in the Supabase database.

### Querying your RAG pipeline assets

Once the pipeline has run successfully, you can query the assets in the Supabase
database using the `--query` flag as well as passing in the model you'd like to
use for the LLM.

When you're ready to make the query, run the following command:

```shell
python run.py --query "how do I use a custom materializer inside my own zenml steps? i.e. how do I set it? inside the @step decorator?" --model=gpt4
```

Alternative options for LLMs to use include:

- `gpt4`
- `gpt35`
- `claude3`
- `claudehaiku`

Note that Claude will require a different API key from Anthropic. See [the
`litellm` docs](https://docs.litellm.ai/docs/providers/anthropic) on how to set
this up.

## ☁️ Running in your own VPC

The basic RAG pipeline will run using a local stack, but if you want to improve
the speed of the embeddings step you might want to consider using a cloud
orchestrator. Please follow the instructions in [our basic cloud setup
guides](https://docs.zenml.io/user-guide/cloud-guide) (currently available for
[AWS](https://docs.zenml.io/user-guide/cloud-guide/aws-guide) and
[GCP](https://docs.zenml.io/user-guide/cloud-guide/gcp-guide)) to learn how you
can run the pipelines on a remote stack.

### BONUS: Connect to ZenML Cloud

If you run the pipeline using ZenML Cloud you'll have access to the managed
dashboard which will allow you to get started quickly. We offer a free trial so
you can try out the platform without any cost. Visit the [ZenML Cloud
dashboard](https://cloud.zenml.io/) to get started.

You can also self-host the ZenML dashboard. Instructions are available in our
[documentation](https://docs.zenml.io/deploying-zenml/zenml-self-hosted).

## 📜 Project Structure

The project loosely follows [the recommended ZenML project structure](https://docs.zenml.io/user-guide/starter-guide/follow-best-practices):

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
│   └── visualise_embeddings.ipynb                      # Notebook to visualize embeddings
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
