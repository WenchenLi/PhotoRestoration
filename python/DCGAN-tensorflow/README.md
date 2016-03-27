Generative Convolutional Net for photo restoration
====================
## IDEA
* Convolution autoencoder/decoder with the help of discriminator

## TODO
* add test sets
* try residual net for generator ( probably not a good idea here, since we prefer remain the original image representation instead of an encoded general or abstract concept).-->
* add batch norm in the restorer
* Check the detail of the implementation 

5. Based on the concept of residual net, add encoded infomation of the problematic image into each layer (maybe face CNN) generator CNN to enforce the CNN to not to be too general/abstract about the input image.
Prerequisites
-------------
- [Tensorflow](https://www.tensorflow.org/)

Usage
-----
First, download dataset with:

    $ mkdir data
    $ python download.py --datasets celebA

To train a model with celebA dataset:

    $ python main.py --dataset celebA --is_train True --is_crop True

To test with an existing model:

    $ python main.py --dataset celebA --is_crop True

Or, you can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    ... add images to data/DATASET_NAME ...
    $ python main.py --dataset DATASET_NAME --is_train True
    $ python main.py --dataset DATASET_NAME

# Results

# Training details

# References
 * [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/Newmu/dcgan_code)
 * [One-Shot Generalization in Deep Generative Models](http://arxiv.org/pdf/1603.05027.pdf)
