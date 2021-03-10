# NESAP Extreme Spatio-Temporal Learning

Deep learning on large spatio-temporal data, including fMRI and climate data.

## Datasets

- Moving MNIST
- Brain fMRI
- Climate

## Models

- PredRNN++

## Package layout

The directory layout of this repo is designed to be flexible:
- Configuration files (in YAML format) go in `configs/`
- Dataset specifications using PyTorch's Dataset API go into `datasets/`
- Model implementations go into `models/`
- Trainer implementations go into `trainers/`. Trainers inherit from
  `BaseTrainer` and are responsible for constructing models as well as training
  and evaluating them.

All examples are run with the generic training script, `train.py`.

## How to run on CPU

To run the examples on the Cori supercomputer, you may use the provided
example SLURM batch script. Here's how to run the Hello World example on 4
Haswell nodes:

`sbatch -N 4 scripts/train_cori.sh configs/hello.yaml`

## How to run on GPU

To run the examples on the Cori supercomputer, you may use the provided
example SLURM batch script. Here's how to run the Hello World example on 4
Haswell nodes:

```{bash}
module load cgpu
sbatch -N 1 scripts/train_prnn3d_fmri.sh
```

## How to get interactive GPU node on Cori
`salloc -N 1 -C gpu -q interactive -t 02:00:00` 

