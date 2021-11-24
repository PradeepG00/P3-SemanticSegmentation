# MSCG-Net for Semantic Segmentation

<!-- TOC -->
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Project File Structure](#project-file-structure)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [How to Train](#how-to-train)
<!-- /TOC -->



## Overview
This repository contains MSCG-Net models (MSCG-Net-50 and MSCG-Net-101) for semantic segmentation in [Agriculture-Vision Challenge and Workshop](https://github.com/SHI-Labs/Agriculture-Vision) (CVPR 2020), and the pipeline of training and testing models, implemented in PyTorch. Please refer to our paper for details:  [Multi-view SelfConstructing Graph Convolutional Networks with Adaptive Class Weighting Loss for Semantic Segmentation](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Liu_Multi-View_Self-Constructing_Graph_Convolutional_Networks_With_Adaptive_Class_Weighting_Loss_CVPRW_2020_paper.pdf)


## Getting Started
### Project File Structure

```
├── checkpoints # output check point, trained weights, log files, tensorboard, etc
├── logs        # model runtime log and function tracing
├── models
├── train.py    # TODO: implement CLI using Click
├── train_R101.py
├── train_R50.py
└── utils       # model block, loss, utils code, # dataset loader and pre-processing code, config code, etc
```

### Dependencies

- python 3.5+
- pytorch 1.4.0
- opencv 3.4+
- tensorboardx 1.9
- albumentations 0.4.0
- pretrainedmodels 0.7.4
- others (see requirements.txt)

### Installation
1. Configure your environment using either `virtual environment`, `anaconda`, or your choice of an environment manager
2. Run the following install the `mscg-net` package dependencies while in the project root directory
```bash
pip install -r requirements.txt # install mscg-models dependencies
pip install -e .  # install mscg-models as a package which resolves the issue of pathing
```

## Usage

### Dataset Preparation
__NOTE__ the current implementation has been hardcoded to support the [2021 dataset](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021) 
1. Change `DATASET_ROOT` to your dataset path in `./data/AgricultureVision/pre_process.py`
```
DATASET_ROOT = '/your/path/to/Agriculture-Vision'
```

2. Keep the dataset structure as the same with the official structure shown as below
```
Agriculture-Vision
|-- train
|   |-- masks
|   |-- labels
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|-- val
|   |-- masks
|   |-- labels
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|-- test
|   |-- boundaries
|   |-- images
|   |   |-- nir
|   |   |-- rgb
|   |-- masks
```

[comment]: <> (#### Dataset Preparation: Known Pitfalls / Errors)

[comment]: <> (1. The current implementation __does NOT properly checks for proper generation of ground-truth__ files)

[comment]: <> (   - A quick-fix for this is remove the  )

### How to Train 
__NOTE__ the current implementation __requires an NVIDIA GPU__   
###  Solution to Memory Issues on a Linux Machine (Ubuntu 20.04)
1.  __IMPORTANT__ Set up the necessary memory to support training __NOTE__ this requires editing the `swap` memory file to allow up to __150gb__ of memory due to the existing implementation
```
# linux
sudo swapoff -a       # disable the current swap memory file
sudo fallocate -l <amount greater than 120>G /swapfile  # specify swap memory size  
sudo chmod 600 /swapfile  # configure user permissions 
sudo mkswap /swapfile   # create the swapfile
sudo swapon /swapfile   # enable the newly created swap memory file
```
2. Run the following to train while inside the project root
```bash
python ./train_R50.py
python ./train_R101.py
```

### Remarks
```
CUDA_VISIBLE_DEVICES=0 python ./tools/train_R50.py  # trained weights checkpoint1
# train_R101.py 								    # trained weights, checkpoint2
# train_R101_k31.py 							    # trained weights, checkpoint3
```

**Please note that:** we first train these models using Adam combined with Lookahead as the optimizer
for the first 10k iterations (around 7~10 epochs) and then change the optimizer to
SGD in the remaining iterations. So you will have to **manually change the code to switch the optimizer**
**to SGD** as follows:  

```
# Change line 48: Copy the file name ('----.pth') of the best checkpoint trained with Adam
train_args.snapshot = '-------.pth'
...
# Comment line 92
# base_optimizer = optim.Adam(params, amsgrad=True)

# uncomment line 93
base_optimizer = optim.SGD(params, momentum=train_args.momentum, nesterov=True)
```

## Test with a single GPU

```
# To reproduce the leaderboard results (0.608), download the trained-weights checkpoint1,2,3
# and save them with the original names into ./checkpoint folder before run test_submission.py
CUDA_VISIBLE_DEVICES=0 python ./tools/test_submission.py
```

#### Trained weights for  3 models download (save them to ./checkpoint before run test_submission)
[checkpoint1](https://drive.google.com/open?id=1eVvUd4TVUtEe_aUgKamUrDdSlrIGHuH3),[checkpoint2](https://drive.google.com/open?id=1vOlS4LfHGnWIUpqTFB2a07ndlpBxFmVE),[checkpoint3](https://drive.google.com/open?id=1nEPjnTlcrzx0FOH__MbP3e_f9PlhjMa2)

## Results Summary

| Models                              | mIoU (%)        | Background      | Cloud shadow    | Double plant    | Planter skip    | Standing water  | Waterway        | Weed cluster    |
| ----------------------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| MSCG-Net-50 (checkpoint1)                 | 54.7            | 78.0            | 50.7            | 46.6            | 34.3            | 68.8            | 51.3            | 53.0            |
| ***MSCG-Net-101 (checkpoint2)***          | ***55.0***      | ***79.8***      | ***44.8***      | ***55.0***      | ***30.5***      | ***65.4***      | ***59.2***      | ***50.6***      |
| MSCG-Net-101_k31 (checkpoint3)            | 54.1            | 79.6            | 46.2            | 54.6            | 9.1             | 74.3            | 62.4            | 52.1            |
| Ensemble_TTA (checkpoint1,2)              | 59.9            | 80.1            | 50.3            | 57.6            | 52.0            | 69.6            | 56.0            | 53.8            |
| <u>**Ensemble_TTA (checkpoint1,2,3)**</u> | 60.8 | 80.5 | <u>**51.0**</u> | 58.6 | 49.8 | <u>**72.0**</u> | 59.8 | <u>**53.8**</u> |
| <u>**Ensemble_TTA (new_5model)**</u> | <u>**62.2**</u> | <u>**80.6**</u> | 48.7 |<u>**62.4**</u> | <u>**58.7**</u> | 71.3| <u>**60.1**</u> | 53.4 |

Please note that all our single model's scores are computed with just single-scale (512x512) and single feed-forward inference without TTA. TTA denotes test time augmentation (e.g. flip and mirror). Ensemble_TTA (checkpoint1,2) denotes two models (checkpoint1, and checkpoint2) ensemble with TTA, and (checkpoint1, 2, 3) denotes three models ensemble. 

### Model Size

| Models           | Backbones           | Parameters | GFLOPs | Inference time <br />(CPU/GPU ) |
| ---------------- | ------------------- | ---------- | ------ | ------------------------------- |
| MSCG-Net-50      | Se_ResNext50_32x4d  | 9.59       | 18.21  | 522 / 26 ms                     |
| MSCG-Net-101     | Se_ResNext101_32x4d | 30.99      | 37.86  | 752 / 45 ms                     |
| MSCG-Net-101_k31 | Se_ResNext101_32x4d | 30.99      | 37.86  | 752 / 45 ms                     |

Please note that all backbones used pretrained weights on **ImageNet** that can be imported and downloaded from the [link](https://github.com/Cadene/pretrained-models.pytorch#senet). And MSCG-Net-101_k31 has exactly the same architecture wit MSCG-Net-101, while it is trained with extra 1/3 validation set (4,431) instead of just using the official training images (12,901). 

## Citation: 
Please consider citing our work if you find the code helps you

[Multi-view Self-Constructing Graph Convolutional Networks with Adaptive Class Weighting Loss for Semantic Segmentation](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Liu_Multi-View_Self-Constructing_Graph_Convolutional_Networks_With_Adaptive_Class_Weighting_Loss_CVPRW_2020_paper.pdf)

```
@InProceedings{Liu_2020_CVPR_Workshops,
author = {Liu, Qinghui and Kampffmeyer, Michael C. and Jenssen, Robert and Salberg, Arnt-Borre},
title = {Multi-View Self-Constructing Graph Convolutional Networks With Adaptive Class Weighting Loss for Semantic Segmentation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```
[Self-Constructing Graph Convolutional Networks for Semantic Labeling](https://arxiv.org/pdf/2003.06932)
```
@inproceedings{liu2020scg,
  title={Self-Constructing Graph Convolutional Networks for Semantic Labeling},
  author={Qinghui Liu and Michael Kampffmeyer and Robert Jenssen and Arnt-Børre Salberg},
  booktitle={Proceedings of IGARSS 2020 - 2020 IEEE International Geoscience and Remote Sensing Symposium},
  year={2020}
}
```
