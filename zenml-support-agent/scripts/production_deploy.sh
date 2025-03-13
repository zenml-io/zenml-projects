#!/bin/bash

# Get the name of the current directory
current_directory=$(basename "$PWD")

# Check if the current directory name is 'llm-agents'
if [ "$current_directory" != "llm-agents" ]; then
  echo "You have to be in the 'llm-agents' directory to run this script."
  exit 1
fi

# Change to src directory
cd src

# Slackbot
echo "Building and deploying slackbot..."
docker build -t europe-west1-docker.pkg.dev/zenml-core/slackbot/zenml-langchain-llamaindex:latest -f Dockerfile-slackbot . --platform linux/amd64
docker push europe-west1-docker.pkg.dev/zenml-core/slackbot/zenml-langchain-llamaindex:latest
gcloud run deploy zenml-langchain --image europe-west1-docker.pkg.dev/zenml-core/slackbot/zenml-langchain-llamaindex:latest --region us-central1

# Web bot
echo "Building and deploying web bot..."
docker build -t europe-west1-docker.pkg.dev/zenml-core/zenbot-website/zenbot-website-lanarky:latest -f zenbot_website.Dockerfile . --platform=linux/amd64
docker push europe-west1-docker.pkg.dev/zenml-core/zenbot-website/zenbot-website-lanarky:latest
gcloud run deploy zenbot-website --image europe-west1-docker.pkg.dev/zenml-core/zenbot-website/zenbot-website-lanarky:latest --region us-central1

echo "Build and deployment complete."
