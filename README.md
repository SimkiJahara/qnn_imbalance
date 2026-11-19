# qnn_imbalance

Class-wise qubit angle analysis for a Quantum Neural Network on CIFAR-10 (2-class).

## Setup

```bash
python -m venv qenv
source qenv/bin/activate
pip install numpy torch torchvision pennylane matplotlib scikit-learn scipy
```

## Data

The CIFAR-10 dataset is **not included** in this repo. It is downloaded
automatically into `./data` the first time you run the script
(`torchvision.datasets.CIFAR10(..., download=True)`), about 163 MB.

## Usage

```bash
python qnn_imbalance.py pairs
python qnn_imbalance.py exp6 --a 0 --b 1 --features resnet --seeds 10
```

Modes: `pairs`, `quick`, `exp2` to `exp7`. Run `python qnn_imbalance.py -h` for all options.
Each run writes to a new subfolder in `results/`.
