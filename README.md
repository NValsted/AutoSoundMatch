# AutoSoundMatch
Experiments regarding automatically estimating synthesizer parameters to reproduce a given sound with support for polyphonic audio signals

# Getting started
A Docker image is provided along with a high level interface to run the experiments.

<b>TLDR</b> Build and run [Docker](https://www.docker.com) image: `make build-image && make run-image-interactive` - then run the experiments with `make mono-benchmark` or `make poly-benchmark`. High-level configuration can be done via the environment variables present in the `Makefile` as well as by applying modifications to the registry, which is done via the `update-registry` command in the `asm-cli.py` CLI, and examples of such are shown in `src/config/fixtures`.

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
The entire library is written in [Python](https://www.python.org/), and dependencies are managed with [Poetry](https://python-poetry.org/). Given Python3.9^ is available, you can install the dependencies with:

```bash
poetry install
```

You may have to run

```bash
poetry env use $PYTHON
```

where `$PYTHON` should be substituted for a Python3.9^-compatible [Python](https://www.python.org/) installation.

The dependencies are available in the `pyproject.toml` file if you wish to use a different dependency manager.

## Run
A [Python](https://www.python.org/)-driven CLI is provided with `asm-cli.py`, which defines useful routines for managing data and training/evaluating models and the NSGA-II algorithm.
An even higher-level interface is provided with make targets in the Makefile. To setup and run the entire experiment in the monophonic setting, simply run the following command from the root of this repo (make sure you've also fetched submodules):

```bash
make mono-benchmark
```

Likewise, run the following to run the entire experiment in the polyphonic setting:

```bash
make poly-benchmark
```

Evaluation results will be stored in an SQLite database which is located at `data/local.db` by default.

### Third-party resources
A number of requried third-party resources are stored remotely. A make target `resources` has been provided to easily download, unpack and store these resources in proper locations. The `mono-benchmark` and `poly-benchmark` targets both invoke `resources` automatically.

### Experiment configurations
Experiments are configurated through parameters to the commands in the `asm-cli.py` CLI as well as through fixtures. A selection of relevant fixtures are provided in the `src/config/fixtures` directory. These are applied using the `update-registry` command, e.g. the following command would apply the MLP fixture, such that an MLP model is trained when the `train-model` command is executed.

```bash
python asm-cli.py update-registry src/config/fixtures/aiflowsynth/mlp.py
```

### Target synth
The setup is fairly synthesizer-agnostic and supports various serialized preset formats, including the standard `.fxp` and `.vstpreset` as well as `.json` files. A relational model of the modulable synthesizer parameters are automatically set up as part of the `setup-relational-models` command, and there is support for <i>locking</i> parameters, which is what is utilized in the documented experiments to limit the input scope of a model to 32 parameters.

A specific setup has been tailored to [Diva](https://u-he.com/products/diva/) by u-he. Using this setup requires manually downloading the synthesizer and pointing the `SYNTH_PATH` environment variable to it. Once that is done, this setup can be triggered in the mono and poly benchmarks by configuring `TARGET_SYNTH=diva`.
