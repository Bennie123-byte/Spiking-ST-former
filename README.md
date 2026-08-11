# Spiking ST-Former

### Enhancing Spatio-Temporal Modeling in Spiking Transformers via Integrated Self-Attention Mechanisms

[![Paper](https://img.shields.io/badge/Paper-The%20Visual%20Computer-2f6f9f.svg)](https://doi.org/10.1007/s00371-025-04173-4)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00371--025--04173--4-blue.svg)](https://doi.org/10.1007/s00371-025-04173-4)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

This repository provides the research code accompanying our paper:

> **Spiking ST-former: enhancing spatio-temporal modeling in spiking transformers via integrated self-attention mechanisms**  
> Yijun Liu, Bin Liu, Wujian Ye, Guoliang Tan, Youfeng Cui  
> *The Visual Computer*, 41(15): 12577–12588, 2025  
> DOI: [10.1007/s00371-025-04173-4](https://doi.org/10.1007/s00371-025-04173-4)

## Overview

Spiking Neural Networks (SNNs) offer an attractive direction for efficient and event-driven artificial intelligence, but many existing Spiking Transformer architectures emphasize spatial interactions while making limited use of the temporal information that is intrinsic to spike-based computation.

**Spiking ST-Former** introduces **Spatio-Temporal Spiking Self-Attention (STSA)** to jointly model spatial and temporal information. A **Temporal Processing Module (TPM)** enables interaction between spike features across adjacent time steps and is integrated into the self-attention process to strengthen temporal representation without introducing an additional parameter-heavy attention branch.

The goal of this project is to improve the accuracy–complexity trade-off of Spiking Transformers and to explore architectures that are better suited to resource-constrained and energy-aware AI systems.

> **Research scope:** the experiments in this repository focus on image and event-based visual recognition. The broader motivation is efficient spatio-temporal modeling for future edge, terminal, and multimodal intelligent systems; this repository does not claim a completed multimodal deployment system.

## Highlights

- **Spatio-Temporal Spiking Self-Attention (STSA)** — extends spatial-only spiking attention with temporal feature interaction.
- **Temporal Processing Module (TPM)** — fuses information across neighboring time steps to better exploit the temporal dynamics of SNNs.
- **Parameter-efficient temporal modeling** — improves temporal representation without adding a separate parameter-intensive temporal attention mechanism.
- **Static and neuromorphic benchmarks** — experiments cover CIFAR-10, CIFAR-100, ImageNet, CIFAR10-DVS, and DVS128 Gesture; the paper also reports N-CALTECH101 results.
- **Open and reproducible research** — training code and configuration files are provided for multiple benchmark datasets.

## Method at a Glance

A conventional spiking self-attention module mainly performs interactions within the spatial/token dimension at each time step. Spiking ST-Former adds temporal interaction before or within attention computation so that the representation at the current time step can incorporate information propagated from previous spike states.

Conceptually:

```text
Spike features over time
        │
        ▼
Temporal Processing Module (TPM)
        │
        ├── cross-timestep feature interaction
        │
        ▼
Spatio-Temporal Spiking Self-Attention (STSA)
        │
        ├── spatial interaction
        └── temporal-aware representation
        │
        ▼
Spiking Transformer encoder
        │
        ▼
Classification head
```

Some source files retain the earlier `TPU` naming used during development; the final paper refers to the module as the **Temporal Processing Module (TPM)**.

## Reported Results

The following classification accuracies are reported in the paper/preprint for representative static and neuromorphic datasets:

| Dataset | Accuracy |
|---|---:|
| CIFAR-10 | **96.15%** |
| CIFAR-100 | **80.44%** |
| CIFAR10-DVS | **82.00%** |
| DVS128 Gesture | **98.74%** |
| N-CALTECH101 | **78.84%** |

ImageNet experiments are also included in the project and publication. Please refer to the published paper for the complete ImageNet model-scale comparison, ablation studies, and parameter/accuracy analysis.

## Repository Structure

The main experiment code is located under `Spiking ST-former/`:

```text
Spiking ST-former/
├── imagenet/          # ImageNet training and evaluation
├── cifar10/           # CIFAR-10 experiments
├── cifar100/          # CIFAR-100 experiments
├── dvs128-gesture/    # DVS128 Gesture experiments
├── cifar10-dvs/       # CIFAR10-DVS experiments
└── TPU/               # Temporal-processing development code
```

## Requirements

Reference software versions used by the original project:

```text
timm==0.6.12
cupy==11.4.0
torch==1.12.1
spikingjelly==0.0.0.0.12
pyyaml
```

Because PyTorch and CuPy builds depend on the CUDA/toolchain environment, please install versions compatible with your local GPU and CUDA setup while keeping the experiment configuration consistent with the versions above when reproducing the original results.

## Training and Evaluation

Clone the repository and enter the experiment directory:

```bash
git clone https://github.com/Bennie123-byte/Spiking-ST-former.git
cd Spiking-ST-former/"Spiking ST-former"
```

### ImageNet

Set hyperparameters in `imagenet/imagenet.yml`.

Training:

```bash
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

Validation/testing:

```bash
cd imagenet
python test.py
```

### CIFAR-10

Set hyperparameters in `cifar10/cifar10.yml`.

```bash
cd cifar10
python train.py
```

### CIFAR-100

Set hyperparameters in `cifar100/cifar100.yml`.

```bash
cd cifar100
python train.py
```

### DVS128 Gesture

```bash
cd dvs128-gesture
python train.py
```

### CIFAR10-DVS

```bash
cd cifar10-dvs
python train.py
```

> Run each command from the `Spiking ST-former/` directory. Dataset locations, distributed-training settings, and other experiment-specific options may need to be adapted to your environment.

## Reproducibility Notes

For reliable comparisons, we recommend recording the following information for each experiment:

- dataset preprocessing and data path;
- random seed;
- time steps and neuron settings;
- model dimension and number of attention heads;
- optimizer, learning-rate schedule, and training epochs;
- CUDA, PyTorch, CuPy, and SpikingJelly versions;
- number and type of GPUs used for distributed training.

Configuration files in each experiment directory should be treated as the starting point for reproducing the reported settings.

## Why Spiking ST-Former Matters

Spiking Transformers combine the representational capacity of Transformer-style architectures with the event-driven temporal dynamics of SNNs. A central challenge is to exploit temporal information without losing the computational advantages that make SNNs attractive in the first place.

Spiking ST-Former studies this problem by integrating temporal processing directly into spiking self-attention. The resulting architecture is intended to provide a better balance between recognition performance and model complexity, which is particularly relevant to research on **efficient AI, neuromorphic computing, edge intelligence, and resource-constrained terminal devices**.

Our longer-term research direction is to connect efficient spiking architectures with more general intelligent systems, including multimodal and hardware-oriented deployment scenarios, while maintaining reproducibility and practical computational efficiency.

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{liu2025spikingstformer,
  title   = {Spiking ST-former: enhancing spatio-temporal modeling in spiking transformers via integrated self-attention mechanisms},
  author  = {Liu, Yijun and Liu, Bin and Ye, Wujian and Tan, Guoliang and Cui, Youfeng},
  journal = {The Visual Computer},
  volume  = {41},
  number  = {15},
  pages   = {12577--12588},
  year    = {2025},
  doi     = {10.1007/s00371-025-04173-4}
}
```

## Acknowledgements

This project builds on and benefits from the following open-source projects:

- [Spikformer](https://github.com/ZK-Zhou/spikformer)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly)

We sincerely thank the authors and maintainers of these projects for making their work available to the research community.

## Maintainer

This repository is actively maintained as part of ongoing graduate research in **Spiking Neural Networks, Spiking Transformers, efficient AI architectures, and hardware-oriented/edge deployment**.

For research questions, reproducibility issues, or collaboration, please use GitHub Issues or contact the authors through the information provided in the paper.
