# Use the zenmldocker/zenml image as the base image
FROM zenmldocker/zenml:0.47.0

# Run 'apt update' to update the package list
RUN apt-get update

# Install curl without any prompts (assume the default 'yes' to all prompts)
RUN apt-get install -y curl

# Download the Git LFS installation script and execute it
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

# Install Git LFS
RUN apt-get install -y git-lfs

# Clear out the local repository of retrieved package files to reduce the image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Initialize Git in the current directory
# Needed to speed up HF push
RUN git init

# Install Git LFS
# Needed to speed up HF push
RUN git lfs install