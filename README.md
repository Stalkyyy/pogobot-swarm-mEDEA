# IAR Project : mEDEA implementation for Pogobot Swarm

## Table of contents

- [IAR Project : mEDEA implementation for Pogobot Swarm](#iar-project--medea-implementation-for-pogobot-swarm)
    - [Table of contents](#table-of-contents)
    - [Overview](#overview)
    - [Dependencies](#dependencies)
    - [Compilation](#compilation)
    - [Usage](#usage)
    - [Documentation](#documentation)

## Overview

This project implements the **mEDEA algorithm** (minimal Environment-Driver Distributed Evolutionary Algorithm) for social learning in a swarm of real Pogobot robots.

The mEDEA algorithm allows each robot to adapt its behavior through interaction with neighboring robots in a decentralized, collective manner. Experiments are conducted on simulated and physical Pogobot-Wheel robots, small wheeled robots capable of exchanging information and moving in a laboratory arena.

## Dependencies

Create a `depencencies` directory at the root of the project :
```
mkdir -p dependencies
cd dependencies
```

Clone the required projects :
- [pogobot-sdk](https://github.com/nekonaute/pogobot-sdk.git)
- [pogosim](https://github.com/Adacoma/pogosim.git)

Follow their Github pages for installation instructions and dependencies.

> If you do not place these projects in `dependencies`, update the Makefile variables :
```
export POGO_SDK=/ABSOLUTE/PATH/TO/pogobot-sdk
export POGOSIM_INCLUDE_DIR=/ABSOLUTE/PATH/TO/pogosim/src
export POGOUTILS_INCLUDE_DIR=/ABSOLUTE/PATH/TO/pogo-utils/src
```

## Compilation

To compile the mEDEA algorithm, run one of the folowwing commands :
```
make clean sim  # To compile the simulation
# OR
make clean bin  # To compile the binary for real Pogobots
# OR
make clean all  # To compile both the simulation and Pogobot binaries
```

### Using Apptainer / Singularity

If you are using Apptainer (or Singularity), the compilation commands should be executed inside the provided container image. For exemple, if the `pogosim.sif` file is in `dependencies/pogosim/` :
```
apptainer exec ./dependencies/pogosim/pogosim.sif make clean sim  # To compile the simulation
# OR
apptainer exec ./dependencies/pogosim/pogosim.sif make clean bin  # To compile the binary for real Pogobots
# OR
apptainer exec ./dependencies/pogosim/pogosim.sif make clean all  # To compile both the simulation and Pogobot binaries
```

> If your Pogobot SDK or Pogosim repositories are not in the `dependencies/` directory, replace the paths above with the paths to your installation :
```
apptainer exec /PATH/TO/pogosim.sif make clean sim
```

## Usage

### Real Pogobots

The folowwing commands illustrate one possible way to deploy the algorithm on real Pogobot robots. For alternative workflows and advanced usage, please refer to the [pogobot-sdk](https://github.com/nekonaute/pogobot-sdk.git) documentation.

```
make connect TTY=/dev/ttyUSB0
```

#### Using Apptainer / Singularity

If you are working inside an Apptainer container, run the command through the .sif image :
```
apptainer exec ./dependencies/pogobot-sdk/pogobot-sdk.sif make connect TTY=/dev/TTYUSB0
```
> Make sure the `TTY` device corresponds to the serial port used by your Pogobot. If the SDK or container image is not located in `dependencies/pogobot-sdk/`, adjust the path accordingly.

### Simulation

Simulation environments are defined in the `conf/` directory. You may modify an existing configuration file or create a new one to define a custom environment.

The following commands shows one possible way to launch the simulation. For more options, refer to the [pogosim](https://github.com/Adacoma/pogosim.git) documentation.

```
./pogobot-swarm-mEDEA -c conf/test.yaml
```
#### Using Apptainer / Singularity

If you are working inside an Apptainer container, run the command through the .sif image :
```
apptainer exec ./dependencies/pogosim/pogosim.sif ./pogobot-swarm-mEDEA -c conf/test.yaml
```
> If the `pogosim.sif` container image is not located in `dependencies/pogosim/`, adjust the path accordingly.

### Analysis scripts (Python)

The analysis and plotting scripts live in `plot_scripts/`. Install their Python dependencies using `requirements.txt` :

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Examples:

- Run multiple seeds and collect metrics into an output folder :
```
python3 plot_scripts/run_collect.py \
  --exe ./pogobot-swarm-mEDEA \
  --configs conf/circle_no_light.yaml conf/square_no_light.yaml \
  --out runs_batch \
  --sim-time 1000 \
  --seeds 0-9
```

- Compare two runs directories and generate plots :
```
python3 plot_scripts/compare_setups.py runs_batch/circle_no_light runs_batch/square_no_light
```

- Additional analysis (distance / orientation plots) :
```
python3 plot_scripts/dist_and_orientation_analysis.py runs_batch/circle_no_light
```

For the gradient light that moves in a orbital pattern around a point with customized velocity, we [forked the pogosim repository and modified it](https://github.com/nekonaute/pogobot-sdk.git). 

## Documentation

The project API and usage instructions are documented in the file [pogodocs.md](pogodocs.md).
