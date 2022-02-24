FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update -yq && apt install software-properties-common -yq
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.9 -yq
RUN ln -s /usr/bin/python3.9 /usr/bin/python

RUN apt-get update -yq && apt-get install -yq \
    curl \
    git \
    make \
    faust \
    build-essential \
    python3.9-dev \
    python3.9-venv \
    python3-setuptools \
    pkg-config \
    mesa-common-dev \
    libasound2-dev \
    libfreetype6-dev \
    libcurl4-gnutls-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    llvm-11 \
    llvm-11-dev \
    libsamplerate0 
RUN apt-get clean -y

# Download/Install poetry
RUN curl -sSL https://install.python-poetry.org | python3.9 -
ENV PATH=/root/.local/bin:$PATH
RUN poetry config virtualenvs.create true
RUN poetry config virtualenvs.in-project true

# Copy current state of this repo and setup Python environment with poetry
COPY . /AutoSoundMatch/
WORKDIR /AutoSoundMatch/
RUN poetry install

# Build DawDreamer
WORKDIR /
RUN git clone --recursive https://github.com/DBraun/DawDreamer.git
RUN ln -s /usr/bin/llvm-config-11 /usr/bin/llvm-config
RUN ln -s /usr/lib/x86_64-linux-gnu/libsamplerate.so.0 /usr/local/lib/libsamplerate.so
ENV CPLUS_INCLUDE_PATH=/usr/include/python3.9/

WORKDIR /DawDreamer/dawdreamer/
RUN git checkout 772048dcbfb2ceb2519d9c6a77917e25970ca1db
RUN git clone https://github.com/grame-cncm/faustlibraries.git

WORKDIR /DawDreamer/Builds/LinuxMakefile/
RUN ldconfig
RUN make CONFIG=Release
WORKDIR /DawDreamer/
RUN python3.9 setup.py install
RUN ln -s /DawDreamer/dawdreamer/dawdreamer.so /AutoSoundMatch/.venv/lib/python3.9/site-packages/

WORKDIR /AutoSoundMatch/
