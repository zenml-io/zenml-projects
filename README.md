<!-- PROJECT LOGO -->
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=03d804c9-e44a-471e-b56d-81085bc925ec" />

<br />
<div align="center">
  <a href="https://zenml.io">
    <img src="_assets/zenml_project.gif" alt="Logo" width="600">
  </a>

<h3 align="center">A home for machine learning projects built
with <a href="https://github.com/zenml-io/zenml/">ZenML</a> and various
integrations.</h3>

  <p align="center">
    Get everything you need to start a project...
    <!-- <div align="center">
      Join our <a href="https://zenml.io/slack" target="_blank">
      <img width="25" src="https://img.shields.io/badge/JOIN US ON SLACK-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Slack"/>
    <b>Slack Community</b> </a> and be part of the ZenML family.
    </div> -->
    <br />
    <a href="https://zenml.io/features">Features</a>
    ·
    <a href="https://zenml.io/roadmap">Roadmap</a>
    ·
    <a href="https://github.com/zenml-io/zenml-projects/issues">Report Bug</a>
    ·
    <a href="https://zenml.io/discussion">Vote New Features</a>
    ·
    <a href="https://blog.zenml.io/">Read Blog</a>
    ·
    <a href="https://zenml.io/meet">Meet the Team</a>
    <br />
    <br /> 
    <a href="https://www.linkedin.com/company/zenml/">
    <img src="https://img.shields.io/badge/JOIN US ON SLACK-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Logo">
    </a>
    <a href="https://www.linkedin.com/company/zenml/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="Logo">
    </a>
    <a href="https://twitter.com/zenml_io">
    <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Logo">
    </a>
    <a href="https://www.youtube.com/c/ZenML">
    <img src="https://img.shields.io/badge/-YouTube-black.svg?style=for-the-badge&logo=youtube&colorB=red" alt="Logo">
    </a>
  </p>
</div>

# ☀️ Introducing ZenML Projects

This repository showcases production-grade ML use cases built with ZenML.
The goal of this repository is to provide you a ready-to-use MLOps workflow that
you can adapt for your application. We maintain a growing list of projects 
from various ML domains including time-series, tabular data, computer vision, 
etc.

# 🧱 Project List

A list of updated and maintained projects by the ZenML team and the community:

| Project                                                                | Tags                                  | Integrations                                                             |
|------------------------------------------------------------------------|---------------------------------------|--------------------------------------------------------------------------|
| [NBA Three-Pointer Predictor](nba-pipeline)                            | Time-series                           | `mlflow` `kubeflow` `evidently` `sklearn` `aws`                          |
| [Time Series Forecasting](time-series-forecast)                        | Time-series                           | `gcp`                                                                    |
| [Customer Satisfaction](customer-satisfaction)                         | Tabular                               | `mlflow` `kubeflow`                                                      |
| [Customer Churn](customer-churn)                                       | Tabular                               | `kubeflow` `seldon`                                                      |
| [Label Studio Annotation](label_studio_annotation)                     | Data Annotation                       | `label-studio`                                                           | 
| [YOLOv5 Object Detection](sign-language-detection-yolov5)              | Computer-vision                       | `mlflow` `gcp`                                                           |
| [LLMs To Analyze Databases](supabase-openai-summary)                   | NLP, LLMs                             | `gcp` `slack`                                                            |
| [GitFlow ZenML Project](https://github.com/zenml-io/zenml-gitflow)     | MLOps with ZenML and GitHub Workflows | `mlflow` `deepchecks` `kserve` `kubeflow` `sklearn` `vertex` `aws` `gcp` |
| [ZenNews](zen-news-summarization)                                      | NLP                                   | `gcp` `vertex` `discord`                                                 |
| [LLM RAG Pipeline with Langchain and OpenAI](llm-agents/)                    | NLP, LLMs                             | `slack` `langchain` `llama_index`                                        |
| [Orbit User Analysis](orbit-user-analysis)                             | Data Analysis, Tabular                | -                                                                        |
| [Huggingface to Sagemaker](huggingface-sagemaker)                      | NLP                                   | `pytorch` `mlflow` `huggingface` `aws` `s3` `kubeflow` `slack` `github`  |
| [Complete Guide to LLMs (from RAG to finetuning)](llm-complete-guide)               | NLP, LLMs                           | `openai` `supabase`  |
| [LLM LoRA Finetuning (Phi3 and Llama 3.1)](llm-lora-finetuning)               | NLP, LLMs                           | `gcp`  |
| [ECP Price Prediction with GCP Cloud Composer](airflow-cloud-composer-etl-feature-train/README.md)               | Regression, Airflow                           | `cloud-composer` `airflow` |
| [Simple LLM finetuning with Lightning Studio](simple-llm-finetuning/README.md)               | Lightning AI Studio, LLMs                           | `cloud-composer` `airflow` |

# 💻 System Requirements

To run any of the projects listed, you have to install ZenML on your machine.
Read [our docs](https://docs.zenml.io/getting-started/installation) for
installation details.

- Linux or macOS.
- Python 3.7, 3.8, 3.9 or 3.10

# 🪃 Contributing

We welcome contributions from anyone to showcase your project built using ZenML.
See our [contributing guide](./CONTRIBUTING.md) to start.

# 🆘 Getting Help

By far the easiest and fastest way to get help is to:

* Ask your questions in [our Slack group](https://zenml.io/slack/).
* [Open an issue](https://github.com/zenml-io/zenml-dashboard/issues/new/choose)
  on our GitHub repo.

# 🔥 About ZenML

ZenML is an extensible, open-source MLOps framework for creating
production-ready ML pipelines. Built for data scientists, it has a simple,
flexible syntax, is cloud- and tool-agnostic, and has interfaces/abstractions
that are catered towards ML workflows.

If you like these projects and want to learn more:

- Give
  the <a href="https://github.com/zenml-io/zenml/stargazers" target="_blank">
  <img width="25" src="https://cdn.iconscout.com/icon/free/png-256/github-153-675523.png" alt="GitHub"/>
  <b>ZenML Repo</b>
  </a> a <b>GitHub Star</b> :star: to show your love!
- Join our <a href="https://zenml.io/slack" target="_blank">
  <img width="25" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
  <b>Slack Community</b>
  </a> and become part of the ZenML family!

# 📜 License

ZenML Projects is distributed under the terms of the Apache License Version 2.0.
A complete version of the license is available in the [LICENSE](LICENSE) file in
this repository. Any contribution made to this project will be licensed under
the Apache License Version 2.0.

# 📖 Learn More

| ZenML Resources             | Description                                                             |
|-----------------------------|-------------------------------------------------------------------------|
| 🧘‍♀️ **[ZenML 101]**       | New to ZenML? Here's everything you need to know!                       |
| ⚛️ **[Core Concepts]**      | Some key terms and concepts we use.                                     |
| 🚀 **[Our latest release]** | New features, bug fixes.                                                |
| 🗳 **[Vote for Features]**  | Pick what we work on next!                                              |
| 📓 **[Docs]**               | Full documentation for creating your own ZenML pipelines.               |
| 📒 **[API Reference]**      | Detailed reference on ZenML's API.                                      |
| 👨‍🍳 **[MLStacks]**        | Terraform-based infrastructure recipes for pre-made ZenML stacks.       |
| ⚽️ **[Examples]**           | Learn best through examples where ZenML is used. We've got you covered. |
| 📬 **[Blog]**               | Use cases of ZenML and technical deep dives on how we built it.         |
| 🔈 **[Podcast]**            | Conversations with leaders in ML, released every 2 weeks.               |
| 💬 **[Join Slack]**         | Need help with your specific use case? Say hi on Slack!                 |
| 🗺 **[Roadmap]**            | See where ZenML is working to build new features.                       |
| 🙋‍♀️ **[Contribute]**      | How to contribute to the ZenML project and code base.                   |

[ZenML 101]: https://docs.zenml.io/user-guide/starter-guide
[Core Concepts]: https://docs.zenml.io/getting-started/core-concepts
[Our latest release]: https://github.com/zenml-io/zenml/releases
[Vote for Features]: https://zenml.io/discussion
[Docs]: https://docs.zenml.io/
[API Reference]: https://apidocs.zenml.io/
[MLStacks]: https://github.com/zenml-io/mlops-stacks
[Examples]: https://github.com/zenml-io/zenml/tree/main/examples
[Blog]: https://blog.zenml.io/
[Podcast]: https://podcast.zenml.io/
[Join Slack]: https://zenml.io/slack-invite/
[Roadmap]: https://zenml.io/roadmap
[Contribute]: https://github.com/zenml-io/zenml/blob/main/CONTRIBUTING.md
