# VIT Tutorial

## Overview

This is my practice repository for the VIT Tutorial, which is a vit classification using hydra and transformers.

## Installation

```bash
conda create -n vit_tutorial python=3.10
conda activate vit_tutorial
pip install torch torchvision torchaudio
pip install hydra-core==1.2.0
pip install transformers==4.20.1
```

## Usage for `hydra_tutorial`

```bash
cd hydra_tutorial
```

```bash
python my_app.py
```


## Usage for `vit_classification`

```bash
cd vit_classification
```

```bash
python main.py
```

Or using hydra to override the config:

```bash
python main.py train.lr=1e-4 data.batch_size=128
```

## Helpful Links

- [Hydra](https://hydra.cc/docs/intro/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [ViT](https://huggingface.co/docs/transformers/model_doc/vit)
- [CIFAR10](https://huggingface.co/datasets/cifar10)
