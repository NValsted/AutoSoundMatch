# AutoSoundMatch
Library for automatically estimating synthesizer parameters to reproduce a given sound with support for polyphonic audio signals

# Getting started
A Docker image is provided along with a high level interface to run the experiments.

<b>TLDR</b> Build and run [Docker](https://www.docker.com) image: `make build-image && make run-image-interactive` - then run the experiments with `make mono-benchmark` or `make poly-benchmark`.


## Environment
### Docker
Given [Docker](https://www.docker.com) is installed, you can run the following command from the root of this repository:

```bash
make build-image
```

and then

```bash
make run-image-interactive
```

to start the container and spawn an interactive shell within it.

The image is based on an NVIDIA CUDA Ubuntu 20.04 base image to allow parallelizing computations on most CUDA-compatible GPUs. Additionally, this requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to be installed on the host machine.

### Alternatives
The entire library is written in Python, and dependencies are managed with [Poetry](https://python-poetry.org/). Given Python3.9^ is available, you can install the dependencies with:

```bash
poetry install
```

You may have to run

```bash
poetry env use $PYTHON
```

where `$PYTHON` should be substituted for a Python3.9^-compatible Python installation.

The dependencies are available in the `pyproject.toml` file if you wish to use a different dependency manager.

## Run
A python-driven CLI is provided with `asm-cli.py`, which defines useful routines for managing data and training/evaluating models and the NSGA-II algorithm.
An even higher-level interface is provided with make targets in the Makefile. To setup and run the entire experiment in the monophonic setting, simply run the following command from the root of this repo:

```bash
make mono-benchmark
```

Likewise, run the following to run the entire experiment in the polyphonic setting:

```bash
make poly-benchmark
```

Evalaution results will be stored in an SQLite database which is located at `data/local.db` by default.

### Third-party resources
A number of requried third-party resources are stored remotely. A make target `resources` has been provided to easily download, unpack and store these resources in proper locations. The `mono-benchmark` and `poly-benchmark` targets both invoke `resources` automatically.

### Experiment configurations
Experiments are configurated through parameters to the commands in the `asm-cli.py` CLI as well as through fixtures. A selection of relevant fixtures are provided in the `src/config/fixtures` directory. These are applied using the `update-registry` command, e.g. the following command would apply the MLP fixture, such that an MLP model is trained when the `train-model` command is executed.

```bash
python asm-cli.py update-registry src/config/fixtures/aiflowsynth/mlp.py
```
