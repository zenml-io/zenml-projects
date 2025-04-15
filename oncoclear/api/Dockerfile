# Use an official Python runtime as a parent image
FROM python:3.9-slim AS base

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain ML libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group and set up a proper home directory
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /home/app && \
    chown -R app:app /home/app
ENV HOME=/home/app

# --- Build Stage --- (Installs dependencies)
FROM base AS builder
COPY requirements.txt .
# Consider using a virtual environment for better isolation
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime Stage --- (Copies app code and sets up execution)
FROM base AS runtime
# Copy installed dependencies from builder stage
# If using venv:
# COPY --from=builder /opt/venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# If not using venv (copying globally installed packages):
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code (main.py, etc.)
# The utils directory should already be in the build context from the deployment step
COPY . .

# Set ZENML_CONFIG_DIR to the home directory
ENV ZENML_CONFIG_DIR=/home/app/.zenml
ENV PYTHONPATH=/app:${PYTHONPATH}

# Ensure directory permissions
RUN mkdir -p $ZENML_CONFIG_DIR && chown -R app:app $ZENML_CONFIG_DIR

# Declare ARG for default port, can be overridden at build time or runtime via ENV
ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

# Switch to the non-root user for security
USER app

# Command to run the application using Uvicorn
# Critical ENV vars (ZENML_*, MODEL_*) must be passed via `docker run -e`
CMD uvicorn main:app --host 0.0.0.0 --port $PORT 