Generative Convolutional Net for photo restoration
====================
## IDEA
* Convolution autoencoder/decoder with the help of discriminator
* We should have this attention mechanism that focusing on generating the missing patch, instead of focusing on the global reconstruction. So maybe send the problematic image to each encoding layer(or even the decoding layer) and finally add the output image filling patch to the problematic image.
* rectangular patch can be used to proximate any problematic area in the input image. So setting different shape of rectangular area as missing patch mask is a good starting point. 
## TODO
* mask shape should be more distinct 
* force attention on filling the problematic area by only backprop the error in the masked area
* rewrite encoder without pretty tensor
* write indicator when to stop training( equilibrium?)
* write residual layer [torch reference](https://github.com/gcr/torch-residual-networks/blob/master/residual-layers.lua)
* add batch norm in the restorer
* try Perceptual Losses from [Perceptual Losses for Real-Time Style Transfer
and Super-Resolution](http://arxiv.org/pdf/1603.08155v1.pdf)

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
