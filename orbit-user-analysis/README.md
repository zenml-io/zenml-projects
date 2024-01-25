# Managing your community with Orbit

Understanding the ZenML community has always been important aspect of our 
journey. It allows us to tailor our framework to meet the needs of our users, 
identify areas for improvement, build relationships with our community, and 
make data-driven decisions.

In order to achieve these goals, we use [Orbit](https://orbit.love/). Orbit is 
a community-building platform that provides a powerful set of tools to help 
you understand and manage your community better. From comprehensive community 
insights to engagement metrics, it helps us to track the growth of our tool 
and take the necessary steps going into the future.

## Project description

In this project, we are using the 
[Orbit API](https://api.orbit.love/reference/about-the-orbit-api) 
to detect members of our community who are either `booming` 
(highly active in the recent period) or `churned`(previously active, but 
recently inactive).

This is achieved by using a ZenML pipeline defined as 
[`community_analysis_pipeline`](pipelines/community_analysis_pipeline.py) 
which features three steps, namely [`booming`](steps/booming.py), 
[`churned`](steps/churned.py), and [`report`](steps/report.py):

- `booming`: detects members who conducted at least **150 events (default)** 
in **the last 14 days (default)**.
- `churned`: detects members who conducted at least one activity in the last 
**6 weeks (default)** but no activity in the **last 2 weeks (default)**.
- `report`: queries the current booming and churned users and saves them as 
output artifacts for tracking purposes. Moreover, it sends a message to a 
selected discord channel. The message includes a URL that directs you to an 
Orbit insight page which displays members from the **last 6 months (default)** 
with the related tags.

# Installation

Install the required packages as follows:

```bash
pip install -r requirements.txt
```

# Secrets

In order to run the pipelines, you need to have a stack with a secret manager
and the following secrets

- `orbit`-`api_token`: the token to use the Orbit API
- `orbit`-`workspace`: the slug of the workspace in Orbit
- `discord`-`webhook_url`: the URL of the webhook that sends messages to a 
specific channel on Discord

# Running it locally

After the installation is completed, you directly run the pipeline locally
right away.

```bash
python run.py
```

# Running it remotely

It is much more ideal to run a pipeline such as the 
`community_analysis_pipeline` on a regular schedule. In order to achieve that, 
you have to [deploy ZenML](https://docs.zenml.io/user-guide/production-guide/deploying-zenml#connecting-to-a-deployed-zenml) 
and set up a stack that supports 
[our scheduling feature](https://docs.zenml.io/user-guide/advanced-guide/pipelining-features/schedule-pipeline-runs). 
Please check [our docs](https://docs.zenml.io/getting-started/introduction) 
for more information.


