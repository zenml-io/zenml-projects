# Magic Photobooth

A personalized AI image generation product that can create your avatars from a photo or selfie.

## ✨ Transform Your Photos into Magical Avatars

Magic Photobooth harnesses cutting-edge AI technology to transform photos into extraordinary digital avatars. By providing a dataset of photos, you can train the AI to generate stunning images in any style, setting, or scenario imaginable.

![Magic Photobooth Sample Gallery](assets/batch-dreambooth.png)

### 🌟 Key Features

- **Personalized AI Models**: Custom-trained on your dataset for truly personalized results
- **Unlimited Creative Possibilities**: Generate images as superheroes, historical figures, or in fantasy worlds
- **Animated Avatars**: Create short animated videos of your digital twin
- **Enterprise-Grade Technology**: Powered by state-of-the-art Flux AI models and ZenML's MLOps framework
- **Fast Processing**: Optimized for quick generation without sacrificing quality

## 🎭 How It Works

Magic Photobooth uses an advanced AI technique called DreamBooth to create a personalized version of powerful image generation models. Here's how the magic happens:

1. **Dataset Preparation**: Prepare a collection of 5-10 clear photos and host them on a compatible storage location (your artifact store).
2. **Configuration**: Update the configuration file with your dataset location and subject details
3. **Model Training**: The system fine-tunes a custom model specifically for your dataset
4. **Avatar Generation**: The system generates images based on customizable text prompts

Behind the scenes, Magic Photobooth employs Low-Rank Adaptation (LoRA) technology to efficiently customize the [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev) image generation model. For inference, we use the optimized [Flux-Schnell](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell) model to deliver high-quality results at impressive speeds.

<video width="512" height="512" controls autoplay loop>
  <source src="assets/hamza_superman.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GPU access (for model training)
- ZenML installed and configured
- Hugging Face account (for hosting your dataset)

### Quick Start

1. Install Magic Photobooth:
   ```bash
   # Clone the repository
   git clone https://github.com/zenml-io/zenml-projects.git
   
   # Navigate to Magic Photobooth
    cd zenml-projects/magic-photobooth
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. Prepare your dataset:
   - Create a collection of 5-10 clear photos
   - Upload them to your artifact store
   - Update the `instance_example_dir` in `configs/k8s_run_refactored_multi_video.yaml` to point to your dataset

3. Customize your subject:
   - Update the `instance_name` and `class_name` in the configuration file to match your subject
   - For example: `instance_name: "john_doe"` and `class_name: "man"`

4. Customize generation prompts:
   - Edit the prompts in the `batch_inference` step in `k8s_run.py` to specify the styles and scenarios you want
   - For example: `"A photo of {instance_phrase} as a superhero"` or `"A photo of {instance_phrase} in ancient Rome"`

5. Explore the interactive tutorial:
   ```bash
   # Open the guided walkthrough notebook
   jupyter notebook walkthrough.ipynb
   ```

## ☁️ Cloud Deployment

Magic Photobooth requires GPU resources for optimal performance. We recommend deploying on a cloud infrastructure:

1. **Set up your cloud environment** using our [1-click deployment guide](https://docs.zenml.io/stacks/deployment/deploy-a-cloud-stack) for AWS, GCP, or Azure.

2. **Configure your GPU quotas** to ensure sufficient resources for model training and inference.

3. **Run the pipeline** using your preferred orchestrator:
   ```bash
   # For Kubernetes environments
   python k8s_run.py
   
   # For Modal
   python modal_run.py
   ```

## 📊 Results & Showcase

Magic Photobooth generates two types of personalized content:

1. **Image Galleries**: Composite images showcasing your avatar in various styles and scenarios
2. **Animated Clips**: 3-second videos bringing your static images to life using [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) technology

## 🛠️ Technical Details

Magic Photobooth is built on a robust MLOps architecture:

```
├── configs/               # Configuration profiles for different environments
├── assets/                # Sample outputs and demonstration media
├── k8s_run.py             # Kubernetes deployment script with customizable prompts
├── modal_run.py           # Modal cloud deployment script
└── walkthrough.ipynb      # Interactive tutorial notebook
```

## 🔮 Use Cases

- **Social Media Content**: Create eye-catching profile pictures and posts
- **Digital Marketing**: Generate custom branded imagery featuring specific individuals
- **Creative Projects**: Visualize subjects in fictional scenarios or historical periods
- **Personal Avatars**: Create unique avatars for gaming or virtual worlds

## 📚 Documentation

For learning more about how to use ZenML to build your own MLOps pipelines, refer to our comprehensive [ZenML documentation](https://docs.zenml.io/).
