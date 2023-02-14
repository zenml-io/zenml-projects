# ZenNews: Generate summarized news on a schedule

Project description

# The goal of the project

The definition of the concrete use-case aside, this project aims to showcase 
some of the advantages that ZenML brings to the table. Some major points we 
would like to highlight include:

- **The ease of use**: ZenML features a simple and clean Python SDK. As you can 
observe in this example, it is not only used to define your steps and 
pipelines but also to access/manage the resources and artifacts that you 
interact with along the way. This makes it significantly easier for our users 
to build their applications around ZenML.
- **The extensibility**: ZenML is an extendable framework. ML projects often 
require custom-tailored solutions and what you get out-of-the-box may not be 
what you need. This is why ZenML is using base abstractions to allow you 
to create your own solutions without reinventing the whole wheel. You can find 
great examples of this feature by taking a look at the custom materializer 
(ArticleMaterializer) and the custom stack component (DiscordAlerter) 
implemented within the context of this project.
- **Stack vs Code**: One of the main features of ZenML is rooted within the 
concept of our stacks. As you follow this example, you will see that there is a 
complete separation between the code and the infrastructure that the pipeline 
is running on. In fact, by utilizing just one flag, it is possible to switch 
from a local default stack to a remote deployment with scheduled pipelines.
- **The scalability**: This is a small PoC-like example which aims to prove 
that ZenML can help you streamline your workflows and accelerate your 
development process. However, it barely scratches the surface of how you can 
improve it even further. For more information, check this section.

# Walkthrough

## Base installation

The `ZenNews` project is designed as a [PyPI package](https://pypi.org/project/zennews/)
that you can install it through `pip`:

```bash
pip install zennews
```

The package comes equipped with the following set of key pieces:

- **The pipeline**: The `zen_news_pipeline` is the main pipeline in this 
workflow. In total, it features three separate steps, namely `collect`, 
`summarize` and `report`. The first step is responsible for collecting 
data, the second step summarizes them and the last step creates a report and 
posts it.
- **The steps**: There is a concrete implementation for each step defined above.
  - For the `collect` step, we have the `bbc_news_source` which (on default) 
  collects the top stories off of the BBC news feed and prepares `Article` 
  objects. 
  - For the `summarize` step, we have implemented `bart_large_cnn_samsum`
  step. As the name suggests, it uses bart model to generate summaries. 
  - Ultimately, for the `report` step, we have implemented the `post_summaries` 
  step. It showcases how a generalized step can function within a ZenML 
  pipeline and uses an alerter to share the results.
- **The materializer**: As mentioned above, the steps within our pipeline are 
using the concept of `Article`s to define their input and output space. Through
this, we can show how to handle the materialization of these artifacts when it 
comes a data type that is not a built-in.
- **The custom stack component**: The ultimate goal of an application such as 
`ZenNews` is to display to outcomes to the user directly. With this project, 
we have used this as a chance to show the extensibility of ZenML in terms of the
stack components and implemented a `DiscordAlerter`.
- **The CLI application**: The example also includes a Click CLI application. 
It utilizes how easily you can use our Python SDK to build your application 
around your ZenML workflows. In order to see it action simply execute:
  ```bash
  zennews --help 
  ```

## Test it locally right away

Once you installed the `zennews` package, you are ready to test it out locally 
right away. The following command will get the top five articles from the BBC
news feed, summarize them and present the results to you. 

> Warning: This will temporarily switch your active ZenML stack to the 
> **default** stack and when the pipeline runs, you will download the model 
> to your local machine.

In order to execute it, simply do:

```bash
zennews bbc
```

You can also parameterize this process. In order to see the possible 
parameters, please use:

```bash
zennews bbc --help
```

## Switching to scheduled pipelines with Vertex

The potential of an application like `ZenNews` can be unlocked by scheduling 
pipelines instead of manually triggering them. In order to showcase it, we 
will set up a fully remote GCP stack and use the `VertexOrchestrator` to 
schedule the pipeline.

### ZenServer

Before you start building the stack, you need to deploy ZenML. For more 
information on how you can achieve that, check 
[the corresponding docs page](https://docs.zenml.io/getting-started/deploying-zenml).

### ZenNews Stack

Once the ZenML is deployed, we can start to build up our stack. Our stack will 
consist of the following components:

- GCP Secrets Manager
- GCP Container Registry
- GCS Artifact Store
- Vertex Orchestrator
- Discord Alerter (part of the zennews package)
 
The first step 
is to install the `gcp` integration:

```bash
zenml integration install gcp
```

### Limitations

- One source
- Schedule cancelling
- Vertex - Schedule
- Alerter (base interface)

# How to contribute?

# Future improvements and upcoming changes

