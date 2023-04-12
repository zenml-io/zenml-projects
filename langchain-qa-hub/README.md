# ZenML Hub Example: QA with LangChain, LLamaIndex and OpenAI

This example demonstrates how to build a question-answering (QA) bot using the ZenML Hub, an open repository for ZenML integrations, plugins, and community-contributed code. The example utilizes the `langchain_qa_example plugin` from the ZenML Hub, which provides a pre-built pipeline for fetching data, creating an index, and answering queries using GPT-3.5 (and beyond) language models powered by OpenAI.

## Getting Started

To run the QA pipeline, follow these steps:

1. Install the `langchain_qa_example` plugin from ZenML Hub using the following command:

```bash
zenml hub install langchain_qa_example
```

2. Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=<YOUR_KEY> # Get it from https://platform.openai.com/account/api-keys
```

3. Import and run the `build_zenml_docs_qa_pipeline` function from the `langchain_qa_example` plugin, passing in the desired question as a parameter. For example:

```python
from zenml.hub.langchain_qa_example import build_zenml_docs_qa_pipeline

# Specify the question to be answered
question = "What is ZenML?"

# Run the QA pipeline
pipeline = build_zenml_docs_qa_pipeline(question=question, load_all_paths=False).run()
```

The pipeline will fetch data from the ZenML documentation, create an index, and use the GPT-3.5 language model to answer the question based on the corpus.

## Customizing the Data Source

The example can also be customized to use a different data source. For instance, you can modify the pipeline to fetch data from a specific URL of your choice by using the `web_url_loader_step` from the `langchain_qa_example` plugin. Here's an example:

```python
from zenml.hub.langchain_qa_example import web_url_loader_step, WebUrlLoaderParameters

# Specify the URLs to fetch data from
urls = ["https://zenml.io/integrations/"]

# Update the pipeline with the custom data loader
qa_pipeline(
    document_loader=web_url_loader_step(WebUrlLoaderParameters(urls=urls)),
    index_generator=index_generator_step(),
    question_answerer=question_answerer_step(
        QAParameters(question="Name five tools that ZenML integrates with.")
    ),
).run()
```

This will fetch data from the specified URLs, create an index, and answer a different question based on the custom data.

For a more thorough walkthrough, see the [Jupyter Notebook version](./langchain-qa-hub.ipynb) of this example.

## Conclusion

The ZenML Hub provides a central location for users to discover, share, and utilize integrations, plugins, and community-contributed code. This example demonstrates how to leverage the `langchain_qa_example` plugin to build a QA bot. Try it out and explore other plugins available in the ZenML Hub to enhance your ML projects!

## Next Steps

The ZenML Hub is open for community contributions and anyone with a GitHub
account can submit plugins. To find out how, check out the
[ZenML Hub Plugin Template](https://github.com/zenml-io/zenml-hub-plugin-template).

If you would like to learn more about the ZenML Hub in general, check out the
[ZenML Hub Documentation](https://docs.zenml.io/collaboration/zenml-hub) or the [ZenML Hub Launch Blog Post](https://blog.zenml.io/zenml-hub-launch).