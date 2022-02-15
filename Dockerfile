FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update -yq && apt-get install -yq \
    wget \
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

# Download/Install Pyflow
RUN wget https://github.com/David-OConnor/pyflow/releases/download/0.3.1/pyflow_0.3.1_amd64.deb
RUN dpkg -i pyflow_0.3.1_amd64.deb

# Copy current state of this repo and setup Python environment with Pyflow
COPY . /AutoSoundMatch/
WORKDIR /AutoSoundMatch/
ENV RUST_BACKTRACE=full
RUN yes 1 | pyflow install

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
RUN ln -s /DawDreamer/dawdreamer/dawdreamer.so /AutoSoundMatch/__pypackages__/3.9/lib/

WORKDIR /AutoSoundMatch/
