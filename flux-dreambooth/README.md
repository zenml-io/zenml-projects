# Flux DreamBooth: Personalized AI with ZenML

This project was motivated by a desire to explore the limits of the latest AI
technologies, particularly the Flux models and the Stable Diffusion img2vid
model. By finetuning the Flux.1-dev model on images of the ZenML co-founder
using DreamBooth, we aimed to showcase the potential for personalized AI
applications.

![](assets/batch-dreambooth.png)

DreamBooth is a technique that allows for the creation of custom text-to-image models by finetuning a pre-trained model on a small dataset of images featuring a specific subject. In this case, we used a dataset of cropped portrait photos of the ZenML co-founder to create a personalized model capable of generating novel images of him in various styles and contexts.

To make the finetuning process more efficient, we employed LoRA (Low-Rank Adaptation) adapters. LoRA allows for faster finetuning by only updating a small fraction of the model's weights, which can then be saved separately from the original model. The resulting LoRA adapters for this project have been pushed to the Hugging Face repository at [https://huggingface.co/strickvl/flux-dreambooth-hamza](https://huggingface.co/strickvl/flux-dreambooth-hamza).

While we finetuned the Flux.1-dev model for training, we switched to the Flux-Schnell model for inference. Flux-Schnell is a faster variant of the Flux model that maintains comparable quality, making it an excellent choice for efficient batch inference. The LoRA adapters worked seamlessly with Flux-Schnell, allowing us to generate high-quality personalized images quickly.

Crafting effective prompts was a crucial aspect of this project. We experimented
with various prompts to find those that worked best with the finetuned model.
Additionally, we had to tailor the prompts to the limitations of our dataset,
which consisted primarily of cropped portrait photos. By carefully designing
prompts that played to the strengths of the available data, we were able to
generate impressive personalized images and short animated videos.

<video width="512" height="512" controls autoplay loop>
  <source src="assets/hamza_superman.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


The pipeline outputs a grouped image showcasing the results of various prompts
and generates short 3-second animated videos using the
[stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
img2vid model. These outputs demonstrate the potential for creating engaging,
personalized content using state-of-the-art AI technologies.

## Getting Started

To get started with this project, follow these steps:

1. Clone the ZenML Projects repository: `git clone https://github.com/zenml-io/zenml-projects.git`
2. Navigate to the `flux-dreambooth` directory: `cd zenml-projects/flux-dreambooth`
3. Install the required dependencies: `pip install -r requirements.txt`
4. For a guided walkthrough of the code, check out the `walkthrough.ipynb` notebook.

## Stack Setup

You'll need a cloud stack to run these pipelines, and you'll need GPU quotas so
as to be able to run the training and batch inference steps. You can get started
with a 1-click deployment on all the major cloud providers using [our simple
guide](https://docs.zenml.io/how-to/stack-deployment/deploy-a-cloud-stack).

## Running the pipeline

To run the pipeline, use one of the following commands depending on your stack:
   - For Kubernetes: `python k8s_run.py`
   - For Modal: `python modal_run.py`

## Project Structure

The project contains the following files with comments:

- `configs/k8s_run_refactored_multi_video.yaml`: Configuration files used in the notebooks
- `k8s_run.py`: Script to run the pipeline on a Kubernetes stack
- `modal_run.py`: Script to run the pipeline on a Modal stack
- `test_examples_utils.py`: Related to the diffusers script used in older versions of the pipeline
- `train_dreambooth_lora_flux.py`: From the diffusers library, used in older versions of the pipeline
- `train_dreambooth.py`: From the diffusers library, used in older versions of the pipeline
- `walkthrough.ipynb`: Jupyter Notebook providing a guided walkthrough of the code

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
