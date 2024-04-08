# Download cuda image
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# To prevent interactive dialogs
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    # Install necessary packages to allow compilation of source code
    && apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    checkinstall \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    software-properties-common \
    # Install python 3.10
    && apt-get update \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
    python3.10

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Install dependencies
COPY requirements.txt ./
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Generate gRPC stub files
COPY *.proto ./
RUN python3.10 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. workerserver.proto

# Copy app files
COPY embeddings ./embeddings
COPY llms ./llms
COPY utils ./utils
COPY app ./app