# scTuner
A repository to easily use or tune single cell models, such as variational autoencoders (VAEs), constructed from large single cell datasets. Additionally, these models can be fine-tuned with smaller datasets, which speeds up the downstream analysis of smaller datasets.

## Model availability
This repository contains its own VAE (constructed with PyTorch). Currently the AdEMAMix Optimiser is implemented from https://arxiv.org/abs/2409.03137.

## Schematic overview of how scTuner can be used
![schematic_plot](img/scTuner_schematic.png)

## Benchmarking training time against state-of-the-art (scVI) integration with scTuner's VAE
![training_plot](img/training_benchmark.png)