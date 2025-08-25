# Adding Projects to ZenML Projects Repository

<!-- Note: This guide is primarily for ZenML internal development use -->

This guide explains how to add your ZenML project to this repository and make it available through the ZenML Projects platform.

## 📋 Requirements

Every project added to this repository must include:

1. **requirements.txt file** - Contains all Python dependencies
2. **Dockerfile** (if special configuration is needed)
3. **Backend configuration** - YAML file in the zenml-projects-backend repository

## 🐳 Docker Configuration

### When to Include a Dockerfile

Include a custom Dockerfile **only if** your project requires:
- Special system dependencies
- Custom environment variables
- Additional configuration beyond Python packages
- Specific build steps

### Dockerfile Structure

If your project needs a custom Dockerfile, follow this exact structure:

```dockerfile
ARG ZENML_VERSION=latest
ARG PROJECT_DIR_NAME
ARG EXTENSION_VERSION=0.1.6

FROM zenmldocker/zenml-codespace:${ZENML_VERSION}

# Set build arguments again for use in subsequent commands
ARG PROJECT_DIR_NAME
ARG EXTENSION_VERSION

# Set the working directory for the project
WORKDIR /home/coder/extensions/zenml-io.zenml-tutorial-${EXTENSION_VERSION}-universal/pipelines

# Copy the specific project's requirements file
COPY ./${PROJECT_DIR_NAME}/requirements.txt /tmp/requirements.txt

# Install project-specific dependencies using uv for faster installation
RUN uv pip install --system --no-cache -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Enable tutorial content for this specific project
ENV ZENML_ENABLE_TUTORIAL=true

# Default command can be overridden in docker run or docker-compose
# CMD ["python", "run.py"] # Or your project's typical entrypoint
```

### When No Dockerfile is Needed

If your project only requires Python dependencies listed in `requirements.txt`, **do not include a Dockerfile**. The projects backend will automatically build your project using the generic Dockerfile available at the zenml-projects-backend repo.

## 🔧 Backend Integration

### YAML Configuration File

Once your project is added to this repository, you **must** create a corresponding YAML configuration file in the `zenml-projects-backend` repository. This file should include all the following attributes:

```yaml
project_id: your-project-id
name: Your Project Name
description: A comprehensive description of what your project does and its key features
github: https://github.com/zenml-io/zenml-projects/tree/main/your-project-directory
preview_image: https://github.com/your-org/your-project/raw/main/assets/preview.png
main_image_link: https://public-flavor-logos.s3.eu-central-1.amazonaws.com/projects/[PROJECT_NUMBER].jpg
details: |
  Detailed description of your project including:
  
  ### What It Does
  
  Explain the main functionality and purpose of your project
  
  ### Key Features
  
  - Feature 1: Description
  - Feature 2: Description
  - Feature 3: Description
  
  ### How It Works
  
  Step-by-step explanation of how users can use your project
  
  ### Architecture
  
  Brief overview of the technical architecture

environment_variables:
  CUSTOM_VAR: 'value'
  ANOTHER_VAR: 'another_value'

stack:
  orchestrator: default
  artifact_store: default
  # Add other stack components if needed

tags:
- domain (e.g., llmops, cv, mlops)
- technology (e.g., pytorch, tensorflow, hugging-face)
- use-case (e.g., classification, forecasting, nlp)

tools_used:
- zenml
- pytorch
- pandas
- scikit-learn
# Add all major tools/libraries used

pipelines:
- name: Main Pipeline Name
  description: Description of what this pipeline does
- name: Secondary Pipeline Name
  description: Description of secondary pipeline (if applicable)

architecture_diagram: assets/architecture-diagram.png  # Optional

codespace:
  enabled: true
  cpu: 4
  memory: 8
```

### Required YAML Attributes

| Attribute | Description | Required |
|-----------|-------------|----------|
| `project_id` | Unique identifier for your project | ✅ |
| `name` | Display name of your project | ✅ |
| `description` | Short description (1-2 sentences) | ✅ |
| `github` | GitHub URL to your project directory | ✅ |
| `preview_image` | URL to preview image | ✅ |
| `main_image_link` | S3 URL to main project image | ✅ |
| `details` | Detailed markdown description | ✅ |
| `environment_variables` | Custom environment variables | ❌ |
| `stack` | ZenML stack configuration | ✅ |
| `tags` | Categorization tags | ✅ |
| `tools_used` | List of technologies used | ✅ |
| `pipelines` | List of pipelines in the project | ✅ |
| `architecture_diagram` | Path to architecture diagram | ❌ |
| `codespace` | Codespace configuration | ❌ |
| `versions` | Version information | ❌ |

## 🖼️ Image Requirements

### Main Project Image

The `main_image_link` should point to an image stored in the S3 bucket:
```
https://public-flavor-logos.s3.eu-central-1.amazonaws.com/projects/[PROJECT_NUMBER].jpg
```

**Image Requirements:**
- Format: JPG or PNG
- Recommended size: 1200x630 pixels
- High quality and representative of your project
- Upload to the S3 bucket before adding to YAML

### Preview Image

The `preview_image` can be:
- Stored in your project's assets directory
- Hosted on GitHub (using raw.githubusercontent.com URLs)
- External hosting (ensure it's reliable)

## 🚀 Deployment Process

### Step-by-Step Process

1. **Add Project to This Repository**
   - Create your project directory
   - Include `requirements.txt`
   - Add `Dockerfile` if needed
   - Include comprehensive README.md
   - Test your project locally

2. **Upload Main Image to S3**
   - Upload your main project image to the S3 bucket
   - Note the project number for the YAML file

3. **Create Backend Configuration**
   - Fork the `zenml-projects-backend` repository
   - Create the YAML configuration file
   - Submit a pull request

4. **Automatic Sync**
   - Once both PRs are merged to main
   - GitHub Actions will automatically sync the project list with Webflow
   - Your project will appear on the ZenML Projects website

### Testing Your Setup

Before submitting your PR, ensure:
- [ ] Your project runs locally with ZenML
- [ ] All dependencies are listed in `requirements.txt`
- [ ] Dockerfile builds successfully (if included)
- [ ] README.md is comprehensive and clear
- [ ] All required YAML attributes are filled out
- [ ] Images are accessible and high quality

## 📝 Example Projects

Refer to existing projects in this repository for examples:
- [ZenML Support Agent](zenml-support-agent/) - LLMOps project
- [Computer Vision End-to-End](end-to-end-computer-vision/) - CV project
- [Credit Scorer](credit-scorer/) - Traditional ML project

## 🆘 Getting Help

If you need help adding your project:
- Join our [Slack community](https://zenml.io/slack)
- Open an issue in this repository
- Check the [ZenML documentation](https://docs.zenml.io/)

## 📜 License

By contributing your project, you agree that it will be licensed under the Apache License Version 2.0, consistent with this repository's license. 