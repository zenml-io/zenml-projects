FROM --platform=linux/amd64 python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libgl1-mesa-dev libgles2-mesa-dev \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# 1) CPU-only torch + pufferlib build deps
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir setuptools wheel Cython numpy

# 2) pufferlib without build isolation (reuses CPU torch from step 1,
#    avoids re-downloading the 916 MB GPU torch in a build sandbox)
RUN pip install --no-cache-dir --no-build-isolation pufferlib

# 3) Remaining runtime deps
RUN pip install --no-cache-dir "zenml[server]" plotly psutil
