# AlexNet Implementation with Pytorch


## 0. Develop Environment
```
numpy == 1.17.2
scipy == 1.3.1
torch == 1.5.1 + cu101
Pillow == 6.2.2
lmdb == 0.97
opencv-python == 3.4.1.15
cryptography == 2.9.2
h5py == 2.7
six == 1.12.0
```
- Pytorch : Stable (1.5) - Linux - Python - CUDA (10.1)
- Using Single GPU (not tested on cpu only)


## 1. Explain about Implementation
- Cause of 32*32 vanish during train and test, resize images to 64*64
- model.py : alexnet
- train.py : train model
- utils.py : count right prediction, save checkpoints


## 2. Brief Summary of *'ImageNet Classification with Deep Convolutional Neural Networks'*

### 2.1. Goal
- Make model with large learning capacity that can learn thousands of objects from million of images
- Make model that have prior knowledge to compensate for all the data we don't have

### 2.2. Intuition
- Powerful models : CNN (convolutional neural networks)
  * Pros
    * Capacity can be controlled by varying their depth and breadth
    * Make strong and mostly correct assumptions about the nature of images
    * Much fewer connections and parameters that makes easier to train, while their theoretically-best performance is likely to be only slightly worse than standard feedforward neural network
  * Cons
    * Expensive to apply in large scale to high-resolution images
- GPU
  * Highly-optimized implementation of 2D convolution
  * Powerful enough to facilitate the training of interestingly-large CNNs
- Large datasets : ImageNet
  * Contain enough labeled examples to train such models without severe overfitting

### 2.3. Dataset
- ILSVRC-2012 dataset
  * train : 1.2 M
  * val : 50 K
  * test : 100 K

### 2.4. AlexNet Configurations
|Layer|Filter Size, Stride, Padding|Input Size|Output Size|
|:-:|:-:|:-:|:-:|
|Conv1|11 * 11, 4, 2|3 * 224 * 224|96 * 55 * 55|
|ReLU|-|-|-|
|LRN|-|-|-|
|Pool1|3 * 3, 2, -|96 * 55 * 55|96 * 27 * 27|
|Conv2|5 * 5, 1, 2|96 * 27 * 27|256 * 27 * 27|
|ReLU|-|-|-|
|LRN|-|-|-|
|Pool2|3 * 3, 2, -|256 * 27 * 27|256 * 13 * 13|
|Conv3|3 * 3, 1, 1|256 * 13 * 13|384 * 13 * 13|
|ReLU|-|-|-|
|Conv4|3 * 3, 1, 1|384 * 13 * 13|384 * 13 * 13|
|ReLU|-|-|-|
|Conv5|3 * 3, 1, 1|384 * 13 * 13|256 * 13 * 13|
|ReLU|-|-|-|
|Pool3|3 * 3, 2, -|256 * 13 * 13|256 * 6 * 6|
|FC1|-|256 * 6 * 6|4096|
|ReLU|-|-|-|
|FC2|-|4096|4096|
|ReLU|-|-|-|
|FC3|-|4096|1000|

- Pooling layer : Max Pooling
- LRN : local response normalization]

### 2.5. Classification Task
#### 2.5.1. Train  
- Data Pre-processing
  * Downsample
    * Rescale image with shorter side of length 256
    * Center crop 256 * 256
  * Random crop 224 * 224
  * Random horizontal flipping
  * Random RGB color shift
  * Normalization : subtract mean RGB value computed on training dataset from each pixel
- Train Details
  * Multinomial logistic regression objective
  * Mini-batch gradient descent based on backpropagation
    * Batch size : 128
    * Learning rate : 0.01
    * Momentum : 0.9
    * L2 weight decay : 0.0005
  * Learning rate scheduler : decrease by a factor of 10 when the validation set accuracy stopped improving
  * Dropout : 0.5 ratio for first 2 FC layer
  * Cycles : 90

#### 2.5.2. Test
- Data Pre-processing
  * Downsample
    * Rescale image with shorter side of length 256
    * Center crop 256 * 256
  * 10-crop : horizontal flip + 5-crop (4 corner + center)
- Test Details
  * No dropout : use all neurons but multiply output by 0.5
- Ensemble
  * Combine the outputs of several models by averaging their soft-max class posteriors
  * Improves the performance due to complementarity of the models


## 3. Reference Paper
- ImageNet Classification with Deep Convolutional Neural Networks [[paper]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
