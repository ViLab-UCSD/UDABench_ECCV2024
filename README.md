## UDA-Bench: Revisiting Common Assumptions in Unsupervised Domain Adaptation Using a Standardized Framework

**ECCV 2024**

<!-- [![PyPI version](https://badge.fury.io/py/udabench.svg)](https://badge.fury.io/py/udabench)
[![Downloads](https://pepy.tech/badge/udabench)](https://pepy.tech/project/udabench) -->

**UDA-Bench** is a comprehensive and standardized PyTorch framework for training and evaluating Unsupervised Domain Adaptation (UDA) methods. This repository provides a foundation for researchers to:

- **Benchmark UDA methods**: Easily compare different UDA techniques on a standardized platform.
- **Implement new UDA methods**: Extend the framework to incorporate novel methods with minimal effort.
- **Reproduce experiments**: Replicate the comparative studies conducted in our ECCV 2024 paper.

**Highlights:**

- **Consolidated Framework**: A unified structure for training and evaluating UDA methods, streamlining the research process.
- **Comprehensive Method Support**: Includes implementations of leading UDA algorithms like DANN, CDAN, MCC, MDD, MemSAC, ILADA, SAFN, BSP, MCD, AdaMatch, DALN, and ToAlign.
- **Modular Architecture**: Easily extendable to accommodate new UDA algorithms, loss functions, and backbones.
- **Reproducibility**: Facilitates replicating experimental setups and results from our paper, enhancing research transparency.

### Installation

Install the necessary dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Datasets

The repository includes dataset loaders for `DomainNet`, `visDA`, `CUB200`, and `OfficeHome`. The train and test files are available in the `data` directory as `.txt` files. You can download the images from the official sources of these datasets.

### UDA Methods

The following UDA methods are currently implemented in this framework:

1. **DANN**
2. **CDAN**
3. **MCC**
4. **MDD**
5. **MemSAC**
6. **ILADA**
7. **SAFN**
8. **BSP**
9. **MCD**
10. **AdaMatch**
11. **DALN**
12. **ToAlign**

### Training

Train an existing UDA method using the following command:

```bash
#!/bin/bash

export trainer=cdan # Select a UDA method
export dataset=DomainNet # Choose a dataset
export n_class=345 # Number of classes
export data_root=/data # Path to data directory
export source=real # Source domain
export target=clipart # Target domain

python3 train.py --config configs/$trainer.yml \
    --source data/$dataset/${source}_train.txt \
    --target data/$dataset/${target}_train.txt \
    --num_class $n_class --data_root $data_root \
    --num_iter 90000 --exp_name test --trainer $trainer
```

**Replace `trainer` with any of the implemented UDA methods.**

### Comparative Studies

Reproduce the comparative studies conducted in our paper:

#### Changing Backbone Architecture

The framework supports the following backbones:

- ResNet50: `resnet50`
- ConvNext: `timm_convnext`
- SWIN: `timm_swin`
- ResMLP: `timm_resmlp`
- DeiT: `timm_deit`

Change the default ResNet-50 architecture using the following command:

```bash
#!/bin/bash

export trainer=cdan
export dataset=DomainNet
export n_class=345
export data_root=/data
export source=real
export target=clipart
export backbone=timm_deit # Select a backbone

python3 train.py --config configs/$trainer.yml \
    --source data/$dataset/${source}_train.txt \
    --target data/$dataset/${target}_train.txt \
    --num_class $n_class --data_root $data_root \
    --num_iter 90000 --exp_name test --trainer $trainer \
    --backbone $backbone
```

#### Effect of Amount of Data

Reproduce results with reduced target unlabeled data:

```bash
#!/bin/bash

export trainer=cdan
export dataset=DomainNet
export n_class=345
export data_root=/data
export source=real
export target=clipart
export backbone=resnet50
export tgt_data_vol=50 # Percentage of target unlabeled data

python3 train.py --config configs/$trainer.yml \
    --source data/$dataset/${source}_train.txt \
    --target data/$dataset/${target}_train.txt \
    --num_class $n_class --data_root $data_root \
    --num_iter 90000 --exp_name test --trainer $trainer \
    --backbone $backbone --target_imb_factor ${tgt_data_vol}
```

### Implementing New UDA Methods

Extend the framework to add new UDA methods:

1. **Add a new config file**: Create `configs/<method>.yaml`.
2. **Implement the forward pass module**: Create `UDA_trainer/<method>.py`.
3. **Implement new loss functions**: Create new loss functions in `losses/`.

You can also modify the architecture, dataloader, or training strategy if needed.

### Citation

If you use this code or our work, please cite our paper:

```text
@article{kalluri2024lagtran,
       author   = {Kalluri, Tarun and Ravichandran, Sreyas and Chandraker, Manmohan},
       title    = {UDA-Bench: Revisiting Common Assumptions in Unsupervised Domain Adaptation Using a Standardized Framework},
       journal  = {ECCV},
       year     = {2024},
       url      = {},
     },
```

### Contact

For any questions or inquiries, please contact [Tarun Kalluri](sskallur@ucsd.edu).