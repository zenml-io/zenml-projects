# ZenNews: Generate summarized news on a schedule

Project description

# The goal of the project

Points we would like to highlight: 

- The ease of use. It has a nice python SDK which would help you build apps
around.
- The same python SDK is extendable you can implement your own logic. There
are two examples of that in this project.
- There is a complete separation between the code and the infra. With one flag 
we can switch between a local test setup to a remote deployment.
- This is a small example that can easily be scaled up.

# Structure of the implementation

## PyPI package

PyPI package explanation

What is included in the package

## Contents

- pipeline
- steps
- custom materializer
- custom stack component alerter
- CLI application

# Walkthrough

## Test case

local execution

## Remote setting: GCP

Remote GCP zenml server

### Requirements

```bash
zenml integration install gcp
```

- Service account
- GCS artifact store
- Vertex Orchestrator
- GCP Secrets Manager
- GCP Container Registry
- Alerter (Optional)

### Limitations

- Vertex - Schedule
- Schedule cancelling
- Alerter (base interface)

# How to contribute?

# Future improvements and upcoming changes

