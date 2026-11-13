# AlexNet 2012 on Tiny ImageNet
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gyrandola/alexnet-tiny-imagenet/blob/main/alexnet_tiny_imagenet.ipynb)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![ROCm 7.2](https://img.shields.io/badge/ROCm-7.2-red.svg)](https://rocm.docs.amd.com/)

An implementation and comparison of the original AlexNet against an adapted version on Tiny ImageNet.

AlexNet was designed for 224x224 input. Applied unchanged to the 64x64 of Tiny ImageNet, the conv trunk
collapses to a 1x1 feature map before the first FC layer, so AdaptiveAvgPool2d((6,6))
broadcasts a single 256-vector 36x into the 9216-dim FC input. Those 36 copies are
rank-redundant: the first Linear(9216, 4096) matrix has rank ≤ 256 at this resolution,
wasting 37M of AlexNet's 60M parameters.

The adaptation reduces the stem's downsampling so the trunk naturally produces the expected 6×6 feature map.

## Model differences

| | OriginalAlexNet | AlexNet (adapted) |
|---|---|---|
| conv1 stride | 4 | 2 |
| pool1 stride | 2 | 1 |
| conv trunk 64x64 | 64->15->7->7->3->3->3->1 | 64->29->27->27->13->13->13->6 |
| conv MACs / image | ~15 M | ~144 M |
| FC MACs / image | ~55 M | ~55 M (same head) |
| total MACs / image |	~70 M|	~199 M |
| params | ~60 M | ~60 M |

## Dataset

Tiny ImageNet (Stanford CS231n), native 64x64, 100,000 train / 10,000 val images,
200 classes.

## Training
The same data-augmentation and training configuration is applied to both models.

- Optimizer: SGD, momentum 0.9, weight decay 5e-4
- LR: 0.02 (linearly scaled from the paper's 0.01 @ 128 to batch 256), cosine annealing over 60 epochs
- Batch size: 256
- Precision: fp16 autocast + GradScaler
- Augmentation: RandomCrop(64, padding=4), horizontal flip, ImageNet mean/std normalization
- Loss: cross-entropy; metrics: top-1 accuracy
- Framework: PyTorch Lightning 2.x, torch.compile, ModelCheckpoint

## Results

<img width="1190" height="390" alt="image" src="https://github.com/user-attachments/assets/8b184e64-f541-454d-b3d3-8594c7bde395" />


| Model | Best val acc | Best val loss | Best train loss |
|---|---:|---:|---:|
| Original | **34.07%** | 2.998 | 0.435 |
| Adapted | **48.01%** | 2.420 | 0.155 | 
| **Δ** | **+13.94 pp** | −0.58 | −0.28 | 

Both models overfit severely. Training accuracy reaches 88% / 95% while validation
plateaus in the high 30s / high 40s. This result is not unexpected for AlexNet applied to small datasets,
especially since batch normalization was not used

The 14 points gap quantifies the cost of the original AlexNet's aggressive downsampling applied to 64×64 input. The trunk collapses to a 1x1 feature map, and AdaptiveAvgPool2d((6,6)) can only broadcast that single vector into the FC input. It cannot recover spatial information the stem already discarded. Reducing the stem stride so the trunk produces a real 6x6 map is worth ~14 pp of validation accuracy for ~3x the total MACs.

This also makes comparisons with newer models on the same dataset fair. The performance gap ResNet18 would show can then be  attributable to BatchNorm and residual connections, not to a flawed feature map at 64x64.

## Requirements
Results were produced in Python 3.14, with `torch==2.14.1+rocm7.2` on AMD RX 9070 XT.
The dataset is downloaded directly by the notebook (240 MB).

## Reference
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). [ImageNet Classification with Deep Convolutional Neural Networks](NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf). *NeurIPS*.
