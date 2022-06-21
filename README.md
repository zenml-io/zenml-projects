# ZenFiles

<div align="center">
    <img src="_assets/zenfiles.png">
</div>

Original Image source: https://www.goodfon.com/wallpaper/x-files-sekretnye-materialy.html
And naturally, all credits to the awesome [X-Files](https://en.wikipedia.org/wiki/The_X-Files) show for this repo's theme!

A collection of `ZenFiles`: Production-grade ML use-cases powered by [ZenML](https://zenml.io/zenml-io/zenml).

# Overview

Here is a list of updated and maintained ZenFiles:

| Name                               | Link                            | Tags                                       | Stack Components                       |
| ---------------------------------- | ------------------------------- | ------------------------------------------ | -------------------------------------- |
| Atari Games Reinforcement Learning | [README](atari-game-play)       | reinforcement-learning                     | default                                |
| Customer Churn                     | [README](customer-churn)        | kubeflow-deployment, seldon-deployment     | kubeflow_orchestrator, seldon_deployer |
| Customer Satisfaction              | [README](customer-satisfaction) | continuous-deployment                      | mlflow_deployer                        |
| NBA Predictor                      | [README](nba-pipeline)          | drift, predictions                         | local_kubeflow_orchestrator            |
| Time Series Forecasting            | [README](time-series-forecast)  | predictions, feature-engineering, vertexai | step_operator, vertex_stack |

# Generate a project template

To generate a ZenFile project folder:

```python
python generate_zenfile.py ZENFILE_NAME  # no requirements needed
```
