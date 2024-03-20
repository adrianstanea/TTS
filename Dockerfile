FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace/

ARG PYTHON_VERSION=3.10

RUN apt-get update -y && \
    apt-get install -y \ 
    build-essential \
    git \
    cmake \
    wget \
    unzip \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* 
# Clean up cache
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create the user
ARG USERNAME=devEnv
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Create virtual environment and add to path (this is equivalent to activating it)
ENV VIRTUAL_ENV=/opt/venv
RUN python${PYTHON_VERSION} -m venv $VIRTUAL_ENV && chmod -R a+rwX $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install requirements
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 install --upgrade pip \
    && pip3 --timeout 900 \
    --trusted-host pypi.org \ 
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --disable-pip-version-check \
    --no-cache-dir \
    install -r /tmp/pip-tmp/requirements.txt \ 
    && rm -rf /tmp/pip-tmp

RUN echo 'alias pip="pip3" ' >> ~/.bashrc && \
    echo 'alias python="python3" ' >> ~/.bashrc