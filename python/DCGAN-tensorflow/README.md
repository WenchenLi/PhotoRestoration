Generative Convolutional Net for photo restoration
====================
## IDEA
* Convolution autoencoder/decoder with the help of discriminator

## TODO
* add batch norm in the restorer
* Check the detail of the implementation 
* try residual net for generator ( maybe not a good idea here, since we prefer remain the original image representation instead of an encoded general or abstract concept).-->
* Based on the concept of residual net, add encoded information of the problematic image into each layer (maybe face CNN) generator CNN to enforce the CNN to not to be too general/abstract about the input image.

Prerequisites
-------------
- [Tensorflow](https://www.tensorflow.org/)

Usage
-----

# Results

# Training details

# References
 * [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/Newmu/dcgan_code)
 * [One-Shot Generalization in Deep Generative Models](http://arxiv.org/pdf/1603.05027.pdf)
 * [Generative Adversarial Networks](http://arxiv.org/abs/1406.2661)
