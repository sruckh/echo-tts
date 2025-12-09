FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System and Python prerequisites (installed before bootstrap runs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        git \
        ca-certificates \
        curl \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && pip install --upgrade pip

WORKDIR /opt/echo-tts
COPY . .

# Ensure bootstrap is executable; runtime can invoke it as needed
RUN chmod +x bootstrap.sh

CMD ["/bin/bash"]
