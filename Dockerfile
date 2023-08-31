FROM nvcr.io/nvidia/pytorch:22.09-py3 as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    ffmpeg \
    libsm6 \
    libxext6 \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/user
WORKDIR /home/user

ENV PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir albumentations==1.3.0 opencv-python==4.5.5.64 imageio==2.9.0 imageio-ffmpeg==0.4.2 \
pytorch-lightning==1.4.2 omegaconf==2.1.1 test-tube>=0.7.5 streamlit==1.12.1 einops==0.3.0 \
transformers==4.25.1 webdataset==0.2.5 kornia==0.6 open_clip_torch==2.0.2 invisible-watermark>=0.1.5 \
streamlit-drawable-canvas==0.8.0 torchmetrics==0.6.0 python-multipart==0.0.5 accelerate==0.16.0 \
google-cloud-storage pytest matplotlib ray[serve]==2.1.0 fastapi==0.85.01 uvicorn[standard]==0.16.0 gpustat==1.0.0

COPY ./backend backend
COPY ./configs configs
COPY ./config.json config.json
RUN pip install -e ./backend/diffusers

WORKDIR ./backend
CMD ["python", "api.py"]
