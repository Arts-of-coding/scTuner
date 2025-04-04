# scTuner
A repository to easily use single cell models (such as VAEs) constructed from large single cell datasets and fine-tune them with smaller datasets.

## Model availability
This repository contains its own VAE (constructed with PyTorch). Currently the AdEMAMix Optimiser is implemented from https://arxiv.org/abs/2409.03137.

## Benchmarking training time against state-of-the-art (scVI) integration with scTuner's VAE
![clustering_plot](img/training_benchmark.png)