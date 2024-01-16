FROM nvidia/cuda:12.0.0-base-ubuntu20.04 as base
RUN apt update

WORKDIR /src

RUN apt-get install -y python3 python3-pip
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install zenml

# create the experiment tracker
RUN zenml experiment-tracker register zenml_wandb --flavor=wandb --api_key=$WANDB_API_KEY
# create the stack with wandb experiment tracker
RUN zenml stack register zencoder_local_stack -o default -a default --experiment_tracker zenml_wandb
# set the new stack 
RUN zenml stack set zencoder_local_stack
# create HF creds
RUN zenml secret create huggingface_creds --username=$HUGGINGFACE_USERNAME --token=$HUGGINGFACE_TOKEN

EXPOSE 8237
EXPOSE 8080

# tart the zenml server
# --port 9000 --ip-address 0.0.0.0
ENTRYPOINT [ "zenml","up", "--port", "8237", "--ip-address", "0.0.0.0"] 



FROM base as deploy
COPY . /src/


ENTRYPOINT [ "python3", "run.py" ]
