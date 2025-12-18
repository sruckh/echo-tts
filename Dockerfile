FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/echo-tts/models/hf-cache \
    HF_HUB_CACHE=/runpod-volume/echo-tts/models/hf-cache

ARG ECHO_TTS_UPSTREAM_REPO="https://github.com/jordandare/echo-tts.git"
ARG ECHO_TTS_UPSTREAM_REF="2ed95fc"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git ca-certificates curl build-essential cmake ninja-build pkg-config ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /opt

# Pin upstream echo-tts code at build time to avoid drift with persistent volumes
RUN git clone "$ECHO_TTS_UPSTREAM_REPO" /opt/echo-tts-remote \
    && cd /opt/echo-tts-remote \
    && git checkout "$ECHO_TTS_UPSTREAM_REF" \
    && sed -i '/gradio/d' requirements.txt \
    && pip install -r requirements.txt \
    && pip install runpod==1.6.1 uvicorn[standard] pydantic python-multipart tqdm boto3

# Copy serverless handler + bootstrap into image
COPY handler.py /opt/echo-tts-remote/handler.py
COPY bootstrap.sh /opt/bootstrap.sh

CMD ["bash", "/opt/bootstrap.sh"]
