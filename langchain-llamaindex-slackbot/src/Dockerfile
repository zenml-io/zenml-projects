FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install Git and any needed packages specified in requirements.txt
RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y git && \
    apt-get clean && \
    pip install -r requirements.txt

COPY . /app

# Make port 3000 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV SLACK_APP_TOKEN $SLACK_APP_TOKEN
ENV SLACK_BOT_TOKEN $SLACK_BOT_TOKEN
ENV OPENAI_API_KEY $OPENAI_API_KEY
ENV PIPELINE_NAME $PIPELINE_NAME
ENV ZENML_SERVER_URL $ZENML_SERVER_URL
ENV ZENML_USERNAME $ZENML_USERNAME
ENV ZENML_PASSWORD $ZENML_PASSWORD
ENV ZENML_ANALYTICS_OPT_IN false

# Run main.py when the container launches and chat_lanarky if arg is passed
CMD ["python", "main.py"]