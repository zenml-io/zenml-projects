FROM zenmldocker/zenml:0.80.0-py3.11

WORKDIR /app

# Pre-install specific versions to fix conflicts
RUN pip install --no-cache-dir kubernetes==21.7.0 botocore==1.37.1 aiobotocore==2.21.1 s3fs==2025.3.1

# Install your pipeline's dependencies
RUN pip install --no-cache-dir polars==1.26.0 textdistance==4.6.3 instructor==1.7.7 \
    jiwer==3.0.5 litellm==1.64.1 openai==1.69.0 mistralai==1.5.0 Pillow==11.1.0

# Set ZenML environment variables
ENV ZENML_LOGGING_COLORS_DISABLED=False