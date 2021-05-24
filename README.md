# AlexNet Implementation with Pytorch
- Unofficial implementation of the paper *ImageNet Classification with Deep Convolutional Neural Networks*


## 0. Develop Environment
```
Docker Image
- tensorflow/tensorflow:tensorflow:2.4.0-gpu-jupyter

Library
- Pytorch : Stable (1.7.1) - Linux - Python - CUDA (11.0)
```
- Using Single GPU


## 1. Implementation Details
- model.py : AlexNet model
- train.py : train AlexNet (include 10-crop on val/test)
- utils.py : count correct prediction
- AlexNet - Cifar 10.ipynb : install library, download dataset, preprocessing, train and result
- Visualize - Kernel.ipynb : visualize the first conv layer
- Details
  * Follow ImaegNet train details : batch size 128, learning rate 0.01, momentum 0.9, weight decay 0.0005
  * No learning rate scheduler for convenience
  * No augmentation using PCA
  * Different network initialization strategy as paper
  * Different image pre-processing as paper (use CIFAR 10 statistics)


## 2. Result Comparison on CIFAR-10
|Source|Score|Detail|
|:-:|:-:|:-|
|Paper|87|without normalization|
|Paper|89|with normalization|
|Current Repo|89.47|with normalization|


## 3. Reference
- ImageNet Classification with Deep Convolutional Neural Networks [[paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
