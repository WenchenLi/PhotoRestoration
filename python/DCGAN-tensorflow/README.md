DCGAN in Tensorflow for photo restoration
====================
Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

*To avoid the fast convergence of D (discriminator) network, G (generatior) network is updatesd twice for each D network update which is a different from original paper.*

## IDEA
1. encode the image using conv net as **input z** to generator, for each layer(or every other layer) of generator encode the input image using convolution and stack together the combined feature maps until it **generating x** (the image send to discriminator)
2. Or encode the problematic image as **conditional embedding y** using convNet first,and use randomnized seed **z**, but still the same stack thing. 


## TODO
<!--* try small size (64 by 64 face) and grey image first 
* define the  loss function of the discriminator such that it is used to differenciate whether it is a good fixation by generator instead of just accept it is a face(since face with a blank area most of the time is still can be seen as a face), refer to [patchMatch](http://gfx.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/patchmatch.pdf) paper regularizer might be helpful.
* try residual net for generator ( probably not a good idea here, since we prefer remain the original image representation instead of an encoded general or abstract concept).-->
1. Find a CNN that is trained on faces with the face recognistion task
2. Take the first FC right after Conv As Z
3. Adjust size of Z as the input to the Generator
4. worry about blending later
5. Based on the concept of residual net, add encoded infomation of the problematic image into each layer (maybe face CNN) generator CNN to enforce the CNN to not to be too general/abstract about the input image.
6. email the possibility of generate faces based on [DRAW: A Recurrent Neural Network For Image Generation](http://arxiv.org/pdf/1502.04623v2.pdf)
7. Original work use cross entropy in the generator, we should use MSE instead [reference](https://www.reddit.com/r/MachineLearning/comments/3klqdh/q_whats_the_difference_between_crossentropy_and/)

Prerequisites
-------------
- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)
- [SciPy](http://www.scipy.org/install.html)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset

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

# Contributors
This work is a forked branch of [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) with good amount of modification for photo restoration.

# References
 * [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/Newmu/dcgan_code)
 * [One-Shot Generalization in Deep Generative Models](http://arxiv.org/pdf/1603.05027.pdf)
