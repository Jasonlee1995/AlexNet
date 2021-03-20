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
- AlexNet - CIFAR 10.ipynb : install library, download dataset, preprocessing, train and result
- Visualize - Kernel.ipynb : visualize the first conv layer
- Details
  * Same batch size, learning rate, learning rate scheduler as paper
  * No LRN on AlexNet model
  * No Augmentation using PCA
  * Different network initialization strategy as paper
  * Different image pre-processing as paper


## 2. Reference
- ImageNet Classification with Deep Convolutional Neural Networks [[paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
