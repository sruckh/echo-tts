FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System and Python prerequisites (installed before bootstrap runs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ffmpeg \
        build-essential \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip

WORKDIR /opt/echo-tts
COPY . .

# Ensure bootstrap is executable; runtime can invoke it as needed
RUN chmod +x bootstrap.sh

CMD ["/bin/bash"]
