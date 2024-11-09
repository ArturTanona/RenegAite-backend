# Build stage
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get -y install \
    python3.10 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    git \
    git-lfs \
    wget

SHELL ["/bin/bash", "-c"]


FROM builder as base
ADD requirements.txt .
RUN pip3 install -r requirements.txt
WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

FROM base as devel
ADD . .
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
