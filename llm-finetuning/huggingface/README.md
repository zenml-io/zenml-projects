# Custom HuggingFace Inference Endpoint Model Deployer

To register a custom huggingface flavor run the following

```bash
zenml init
zenml stack register zencoder_hf_stack -o default -a default
zenml stack set zencoder_hf_stack
export HUGGINGFACE_USERNAME=<here>
export HUGGINGFACE_TOKEN=<here>
zenml secret create huggingface_creds --username=$HUGGINGFACE_USERNAME --token=$HUGGINGFACE_TOKEN
zenml model-deployer flavor register huggingface.hf_model_deployer_flavor.HuggingFaceModelDeployerFlavor
```

Afterward, you should see the new flavor in the list of available flavors:

```bash
zenml model-deployer flavor list
```

Register model deployer component into the current stack

```bash
zenml model-deployer register hfendpoint --flavor=hfendpoint --token=$HUGGINGFACE_TOKEN
zenml stack update zencoder_hf_stack -d hfendpoint
```
