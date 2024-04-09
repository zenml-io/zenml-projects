# 🦜 Production-ready RAG pipelines for chat applications

This project showcases how you can work up from a simple RAG pipeline to a more complex setup that
involves finetuning embeddings, reranking retrieved documents, and even finetuning the
LLM itself. We'll do this all for a use case relevant to ZenML: a question
answering system that can provide answers to common questions about ZenML. This
will help you understand how to apply the concepts covered in this guide to your
own projects.

![](.assets/rag-pipeline-zenml-cloud.png)

Contained within this project is all the code needed to run the full pipelines.
You can follow along [in our guide](https://docs.zenml.io/user-guide/llmops-guide/) to understand the decisions and tradeoffs
behind the pipeline and step code contained here. You'll build a solid understanding of how to leverage
LLMs in your MLOps workflows using ZenML, enabling you to build powerful,
scalable, and maintainable LLM-powered applications.

This project contains all the pipeline and step code necessary to follow along
with the guide. You'll need a PostgreSQL database to store the embeddings; full
instructions are provided below for how to set that up.

## 🙏🏻 Inspiration and Credit

The RAG pipeline relies on code from [this Timescale
blog](https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector/)
that showcased using PostgreSQL as a vector database. We adapted it for our use
case and adapted it to work with Supabase.

## 🏃 How to run

This project showcases production-ready pipelines so we use some cloud
infrastructure to manage the assets. You can run the pipelines locally using a
local PostgreSQL database, but we encourage you to use a cloud database for
production use cases.

### Connecting to ZenML Cloud

If you run the pipeline using ZenML Cloud you'll have access to the managed
dashboard which will allow you to get started quickly. We offer a free trial so
you can try out the platform without any cost. Visit the [ZenML Cloud
dashboard](https://cloud.zenml.io/) to get started.

### Setting up Supabase

[Supabase](https://supabase.com/) is a cloud provider that provides a PostgreSQL database. It's simple to
use and has a free tier that should be sufficient for this project. Once you've
created a Supabase account and organisation, you'll need to create a new
project.

![](.assets/supabase-create-project.png)

You'll then want to connect to this database instance by getting the connection
string from the Supabase dashboard.

![](.assets/supabase-connection-string.png)

You'll then use these details to populate some environment variables where the pipeline code expects them:

```shell
export ZENML_SUPABASE_USER=<your-supabase-user>
export ZENML_SUPABASE_HOST=<your-supabase-host>
export ZENML_SUPABASE_PORT=<your-supabase-port>
```

You'll want to save the Supabase database password as a ZenML secret so that it
isn't stored in plaintext. You can do this by running the following command:

```shell
zenml secret create supabase_postgres_db --password="YOUR_PASSWORD"
```

### Running the RAG pipeline

To run the pipeline, you can use the `run.py` script. This script will allow you
to run the pipelines in the correct order. You can run the script with the
following command:

```shell
python run.py --basic-rag
```

This will run the basic RAG pipeline, which scrapes the ZenML documentation and stores the embeddings in the Supabase database.

### Querying your RAG pipeline assets

Once the pipeline has run successfully, you can query the assets in the Supabase
database using the `--rag-query` flag as well as passing in the model you'd like
to use for the LLM.

In order to use the default LLM for this query, you'll need an account
and an API key from OpenAI specified as another environment variable:

```shell
export OPENAI_API_KEY=<your-openai-api-key>
```

When you're ready to make the query, run the following command:

```shell
python run.py --rag-query "how do I use a custom materializer inside my own zenml steps? i.e. how do I set it? inside the @step decorator?" --model=gpt4
```

Alternative options for LLMs to use include:

- `gpt4`
- `gpt35`
- `claude3`
- `claudehaiku`

Note that Claude will require a different API key from Anthropic. See [the
`litellm` docs](https://docs.litellm.ai/docs/providers/anthropic) on how to set this up.

## ☁️ Running with a remote stack

The basic RAG pipeline will run using a local stack, but if you want to improve
the speed of the embeddings step you might want to consider using a cloud
orchestrator. Please follow the instructions in [our basic cloud setup guides](https://docs.zenml.io/user-guide/cloud-guide)
(currently available for [AWS](https://docs.zenml.io/user-guide/cloud-guide/aws-guide) and [GCP](https://docs.zenml.io/user-guide/cloud-guide/gcp-guide)) to learn how you can run the pipelines on
a remote stack.

## 📜 Project Structure

The project loosely follows [the recommended ZenML project structure](https://docs.zenml.io/user-guide/starter-guide/follow-best-practices):

```
.
├── LICENSE                                             # License file
├── README.md                                           # This file
├── constants.py                                        # Constants for the project
├── pipelines
│   ├── __init__.py                                    
│   └── llm_basic_rag.py                                # Basic RAG pipeline
├── requirements.txt                                    # Requirements file
├── run.py                                              # Script to run the pipelines
├── steps
│   ├── __init__.py                                     
│   ├── populate_index.py                               # Step to populate the index
│   ├── url_scraper.py                                  # Step to scrape the URLs
│   ├── url_scraping_utils.py                           # Utilities for the URL scraper
│   └── web_url_loader.py                               # Step to load the URLs
└── utils                                              
    ├── __init__.py
    └── llm_utils.py                                    # Utilities related to the LLM
```
