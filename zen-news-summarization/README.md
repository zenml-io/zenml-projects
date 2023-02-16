# 📜 ZenNews: Generate summarized news on a schedule

In today's information age, we are bombarded with a constant stream of news 
and media from a variety of sources. Summarizing tasks, particularly when it 
comes to news sources, can be a powerful tool for the efficient consumption of 
information. They distill complex or lengthy content into easily 
digestible chunks that can be scanned and absorbed quickly, allowing us to 
keep up with the news without being overwhelmed. They can also help us separate 
the signal from the noise, highlighting the most important details and helping 
us identify what's worth further investigation. 

This is where **ZenNews** come into play. It offers a tool that can 
automate the summarization process and save users time and effort while 
providing them with the information they need. This can be particularly valuable 
for busy professionals or anyone who wants to keep up with the news but doesn't 
have the time to read every article in full.

# 🎯 The goal of the project

The definition of the concrete use case aside, this project aims to showcase 
some of the advantages that ZenML brings to the table. Some major points we 
would like to highlight include:

- **The ease of use**: ZenML features [a simple and clean Python 
SDK](https://docs.zenml.io/starter-guide/pipelines). As you can 
observe in this example, it is not only used to define your steps and 
pipelines but also to access/manage the resources and artifacts that you 
interact with along the way. This makes it significantly easier for our users 
to build their applications around ZenML.

- **The extensibility**: ZenML is an extendable framework. ML projects often 
require custom-tailored solutions and what you get out of the box may not be 
what you need. This is why ZenML is using base abstractions to allow you 
to create your own solutions without reinventing the whole wheel. You can find 
great examples of this feature by taking a look at the custom materializer 
([ArticleMaterializer](src/zennews/materializers/article_materializer.py)) 
and the custom stack component 
([DiscordAlerter](src/zennews/alerter/discord_alerter.py)) 
implemented within the context of this project.

- **Stack vs Code**: One of the main features of ZenML is rooted within the 
concept of our stacks. As you follow this example, you will see that there is a 
complete separation between the code and the infrastructure that the pipeline 
is running on. In fact, by utilizing just one flag, it is possible to switch 
from a local default stack to a remote deployment with scheduled pipelines.

- **The scalability**: This is a small PoC-like example that aims to prove 
that ZenML can help you streamline your workflows and accelerate your 
development process. However, it barely scratches the surface of how you can 
improve it even further. For more information, check 
[this section](#-limitations-future-improvements-and-upcoming-changes).

# 🐍 Base installation

The **ZenNews** project is designed as a 
[PyPI package](https://pypi.org/project/zennews/)
that you can install through `pip`:

```bash
pip install zennews
```

The package comes equipped with the following set of **key** pieces:

- **The pipeline**: The [`zen_news_pipeline`](src/zennews/pipelines/zen_news_pipeline.py) 
is the main pipeline in this workflow. In total, it features three steps, 
namely `collect`, `summarize` and `report`. The first step is responsible 
for collecting articles, the second step summarizes them, and the last step 
creates a report and posts it.
- **The steps**: There is a concrete implementation for each step defined above.
  - For the `collect` step, we have the [`bbc_news_source`](src/zennews/steps/sources/bbc.py)
  which (on default) collects the top stories from the BBC news feed and 
  prepares [`Article`](src/zennews/models/article.py) objects. 
  - For the `summarize` step, we have implemented [`bart_large_cnn_samsum`](src/zennews/steps/summarize/bart_large_cnn_samsum.py)
  step. As the name suggests, it uses bart model to generate summaries. 
  - Ultimately, for the `report` step, we have implemented the 
  [`post_summaries`](src/zennews/steps/report/report.py) 
  step. It showcases how a generalized step can function within a ZenML 
  pipeline and uses an alerter to share the results.
- **The materializer**: As mentioned above, the steps within our pipeline are 
using the concept of `Article`s to define their input and output space. Using 
the [`ArticleMaterializer`](src/zennews/materializers/article_materializer.py), 
we can show how to handle the materialization of these artifacts 
when it comes to a data type that is not built-in.
- **The custom stack component**: The ultimate goal of `ZenNews` is to 
serve the use the direct outcomes of the pipeline. That is why we have used it
as a chance to show the extensibility of ZenML in terms of the stack components 
and implemented a [`DiscordAlerter`](src/zennews/alerter/discord_alerter.py).
- **The CLI application**: The example also includes a 
[Click](https://click.palletsprojects.com/en/8.1.x/) CLI application. 
It utilizes how easily you can use our Python SDK to build your application 
around your ZenML workflows. In order to see it in action simply execute:

  ```bash
  zennews --help 
   ```

# 🕹 Test it locally right away

Once you installed the `zennews` package, you are ready to test it out locally 
right away. The following command will get the top five articles from the BBC
news feed, summarize them and present the results to you. 

> Warning: This will temporarily switch your active ZenML stack to the 
> **default** stack and when the pipeline runs, you will download the model 
> to your local machine.

```bash
zennews bbc
```

You can also parameterize this process. In order to see the possible 
parameters, please use:

```bash
zennews bbc --help
```

# 🚀 Switching to scheduled pipelines with Vertex

The potential of an application like `zennews` can be only unlocked by 
scheduling summarization pipelines instead of manually triggering them 
yourself. In order to showcase it, we will set up a fully remote GCP stack 
and use the `VertexOrchestrator` to schedule the pipeline.

## Deploy ZenML on GCP

Before you start building the stack, you need to deploy ZenML on GCP. For more 
information on how you can achieve do that, please check 
[the corresponding docs page](https://docs.zenml.io/getting-started/deploying-zenml).

## ZenNews Stack

Once the ZenML is deployed, we can start to build up our stack. Our stack will 
consist of the following components:

- [GCP Secrets Manager](https://docs.zenml.io/component-gallery/secrets-managers/gcp)
- [GCP Container Registry](https://docs.zenml.io/component-gallery/container-registries/gcloud)
- [GCS Artifact Store](https://docs.zenml.io/component-gallery/artifact-stores/gcloud-gcs)
- [Vertex Orchestrator](https://docs.zenml.io/component-gallery/orchestrators/gcloud-vertexai)
- [Discord Alerter (part of the `zennews` package)](src/zennews/alerter/discord_alerter.py)
 
Let's start by installing the `gcp` integration:

```bash
zenml integration install gcp
```

### Secrets Manager

The first component to register is a
[GCP secrets manager](https://docs.zenml.io/component-gallery/secrets-managers/gcp). 
The command is quite straightforward. You just have to give it a name and 
provide the ID of your project on GCP.

```bash
zenml secrets-manager register <SECRETS_MANAGER_NAME> \
    --flavor=gcp \
    --project_id=<PROJECT_ID>
```

### Container Registry

The second component is a 
[GCP container registry](https://docs.zenml.io/component-gallery/container-registries/gcloud). 
Similar to the previous component, you just need to provide a name and the 
URI to your container registry on GCP.

```bash
zenml container-registry register <CONTAINER_REGISTERY_NAME> \
    --flavor=gcp \
    --uri=<REGISTRY_URI>
```

### Artifact Store

The next component on the list is a 
[GCS artifact store](https://docs.zenml.io/component-gallery/artifact-stores/gcloud-gcs). 
In order to register it, all you have to do is to provide the path to your GCS
bucket:

```bash 
zenml artifact-store register <ARTIFACT_STORE_NAME> \
    --flavor=gcp \
    --path=<PATH_TO_BUCKET> 
```

### Orchestrator

Following the artifact store, we will register a
[Vertex AI orchestrator.](https://docs.zenml.io/component-gallery/orchestrators/gcloud-vertexai)

```bash
zenml orchestrator register <ORCHESTRATOR_NAME> \
    --flavor=vertex \
    --project=<PROJECT_ID> \
    --location=<GCP_LOCATION> \
    --workload_service_account=<EMAIL_OF_YOUR_SERVICE_ACCOUNT> \
    --service_account_path=<PATH_TO_YOUR_SERVICE_ACCOUNT_KEY>
```

You need to simply provide the id of your project, the name of your GCP 
region and the service account you would like to use.

> Warning: In this version, you have to provide both the email of the service 
> account and the path to a key.json file. This interaction will be improved 
> with the upcoming releases.

Make sure that the service account has the proper roles for the following 
services: Cloud Functions, Cloud Scheduler, Secret Manager, Service Account,
Storage, and Vertex AI,


### GCP Stack

With these four components, we are ready to establish and activate the base 
version of our GCP stack.

```bash
zenml stack register <STACK_NAME> \
    -x <SECRETS_MANAGER_NAME> \
    -c <CONTAINER_REGISTERY_NAME> \
    -a <ARTIFACT_STORE_NAME> \
    -o <ORCHESTRATOR_NAME> \
    --set
```

### Alerter 

The last component in our stack is a special case. As mentioned before
the `zennews` package already comes equipped with a custom stack component 
implementation, namely the `DiscordAlerter`. In a nutshell, it uses the 
[**discord.py**](https://discordpy.readthedocs.io/en/stable/index.html) package
to send messages via a webhook to a discord text channel. You can find the 
implementation right [here](src/zennews/alerter/discord_alerter.py).

The following sections show how we can register `DiscordAlerter` as a custom 
flavor , create an instance of it, and update our stack.

#### Registering the custom flavor

 All you 
have to do to register such custom flavor is to provide the corresponding
source path to the flavor class.

```bash
zenml alerter flavor register zennews.alerter.discord_alerter_flavor.DiscordAlerterFlavor
```

ZenML will import and add that to the list of available alerter flavors.

```bash 
zenml alerter flavor list
```

#### Registering the alerter

Now that the flavor is registered, you can create an alerter with the flavor
`discord-webhook`. Through this example, you will also see how you can use 
secret references to handle sensitive information during the registration of 
stack components.

Let's start by registering the secret:

```bash
zenml secrets-manager secret register <SECRET_NAME> \
    --webhook_url=<ACTUAL_URL_OF_THE_WEBHOOK>
```

This will use the secrets manager in our active GCP stack. Once the secret 
registration is complete, you can register your alerter as follows:

```bash
zenml alerter register <ALERTER_NAME> \
    --flavor discord-webhook \
    --webhook_url=<SECRET_REFERENCE>  # formatted as {{SECRET_NAME:WEBHOOK_URL}}
```

#### Updating the stack

The last step is to update our stack with our new alerter:

```bash
zenml stack update <STACK_NAME> -al <ALERTER_NAME>
```

## Scheduling pipelines through the `zennews` CLI

Now the stack is set up, you can use the `--schedule` option when you run your 
`zennews` pipeline. There are three possible values that you can use for the 
`schedule` option: `hourly`, `daily` (every day at 9 AM), or `weekly` (every
Monday at 9 AM).

```bash
zennews bbc --schedule daily
```

This will use your active stack (the GCP stack) and schedule your ZenNews 
pipeline.

# 📙 Limitations, future improvements and upcoming changes

Before we end this project, it is also important to talk about the limitations
we faced, the possible future improvements, and changes that are already in 
motion:

- The first limitation of ZenNews is the number of supported news sources.
As this project was initially designed as a PoC, the only supported news 
source is BBC. However, thanks to our design, it is really easy to expand this 
list by adding additional steps, which consume data and create `Article` 
objects.
- The ability to schedule pipelines through ZenML played a critical role 
within the context of this project. However, this feature has its own 
limitations. While you can create scheduled pipelines, once the pipeline and 
its schedule is created, you can not cancel or modify the behaviour of this 
scheduled pipeline. This means, if you want to cancel it, you have to do it 
over the orchestrator yourself.
- The other limitation regarding the schedules is the format. As of now, the 
CLI application takes the user input and converts it into a cron expression.
Any orchestrator which does not support these expressions will not applicable.
- As the ZenML team, we have been looking for ways to improve the interface 
of our base alerters. You might see some changes in the upcoming releases. 
- Similar to the alerters, we are working on improving the management of our 
secrets. 
 
Tune in to [our slack](https://zenml.io/slack-invite/) to stay updated about 
the upcoming changes and ask any questions you might have.
