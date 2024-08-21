## UDA-Bench: Revisiting Common Assumptions in Unsupervised Domain Adaptation Using a Standardized Framework
### ECCV 2024.

## UDABench

We design a new consolidated framework for standardized training and evaluation of UDA methods in PyTorch. We give a brief overview of the training existing UDA implementations, along with guidance on implementaing new UDA methods using our framework.

### Datasets

The train and test files for all the datasets used in the paper are present in `data` as `.txt` files. The repository currently contains datasets for `DomainNet`, `visDA`, `CUB200` and `OfficeHome`. 

The images can be downloaded from the official sources of these datasets.

### UDA Methods.

The following UDA methods have been currently implemented in this framework.

 1. DANN
 2. CDAN
 3. MCC
 4. MDD
 5. MemSAC
 6. ILADA
 7. SAFN
 8. BSP
 9. MCD
 10. AdaMatch 
 11. DALN 
 12. ToAlign

### Requirements

You can install the necessary dependencies required for the framework using the `requirements.txt` file as follows:

```
pip install -r requirements.txt
```

### Training

For training one of the existing UDA methods, you can use the following command.

```
#!/bin/bash

export trainer=cdan
export dataset=DomainNet
export n_class=345
export data_root=/data
export source=real
export target=clipart

python3 train.py --config configs/$trainer.yml --source data/$dataset/${source}_train.txt --target data/$dataset/${target}_train.txt --num_class $n_class --data_root $data_root --num_iter 90000 --exp_name test --trainer $trainer
```

You can replace the `trainer` using any of the implemented UDA methods.


### Comparative Studies

We additionally facilitate easy reproduction of the comparative studies conducted in the paper. You can use the following modifications to the command above to run these experiments.

#### Changing backbone architecture

The framework currently supports the following backbones.

 - ResNet50: `resnet50`
 - ConvNext: `timm_convnext`
 - SWIN: `timm_swin`
 - ResMLP: `timm_resmlp`
 - DeiT: `timm_deit`

To change the default ResNet-50 architecture, you can use the following command.

```
#!/bin/bash

export trainer=cdan
export dataset=DomainNet
export n_class=345
export data_root=/data
export source=real
export target=clipart
export backbone=timm_deit

python3 train.py --config configs/$trainer.yml --source data/$dataset/${source}_train.txt --target data/$dataset/${target}_train.txt --num_class $n_class --data_root $data_root --num_iter 90000 --exp_name test --trainer $trainer --backbone $backbone
```

#### Effect of amount of Data

To reproduce the results with reduced target unlabeled data, you can use the following command. 

```
#!/bin/bash

export trainer=cdan
export dataset=DomainNet
export n_class=345
export data_root=/data
export source=real
export target=clipart
export backbone=resnet50
export tgt_data_vol=50

python3 train.py --config configs/$trainer.yml --source data/$dataset/${source}_train.txt --target data/$dataset/${target}_train.txt --num_class $n_class --data_root $data_root --num_iter 90000 --exp_name test --trainer $trainer --backbone $backbone --target_imb_factor ${tgt_data_vol}
```

The `tgt_data_vol` should be a number indicating the percentage of target unlabeled data to be used. By default, the class-balanced sub-sampling strategy will be applied.

### Implementing new UDA methods.


The framework can easily be extended to add newer UDA methods. For this, the following modifications can be performed.

1. Add a new config file in `configs/<method>.yaml`
2. Implement the forward pass module in `UDA_trainer/<method>.py`.
3. Implement new loss functions in `losses/`.

The architecture, dataloader or the training strategy can also be modified if necessary.

### Citation

If this code or our work helps in your work, please consider citing us. 
``` text
@article{kalluri2024lagtran,
        author    = {Kalluri, Tarun and Ravichandran, Sreyas and Chandraker, Manmohan},
        title     = {UDA-Bench: Revisiting Common Assumptions in Unsupervised Domain Adaptation Using a Standardized Framework},
        journal   = {ECCV},
        year      = {2024},
        url       = {},
      },
```

### Contact

If you have any question about this project, please contact [Tarun Kalluri](sskallur@ucsd.edu).